---
layout: default
title:  "Predicting Streamer Wins Using Basic Computer Vision"
date:   2020-12-13 12:55:30 -0500
categories: computer_vision
---

## Introduction
I'm an avid follower of streamer and YouTuber [Northernlion](https://www.youtube.com/user/Northernlion)(NL) and so are thousands of other people. So much so, that people have manually collected a ton of data on how he performs when playing his most popular game, The Binding of Isaac. I thought it would be cool if we could predict whether he would win or lose his run as he's playing, and even cooler if we could augment the data source with additional information automatically through image analysis using the frames of the video.

## Getting the Data
Over at [Northernlion-db](https://www.northernlion-db.com/) they've done an incredible job manually collecting data on each run NL has done in the game and they even provide a Postgres dump of that data. So I downloaded that and setup a Postgres instance on my local machine so I could easily explore and query the info. The data isn't super clean but also not terrible. Most of the work was extracting encoded data into additional features and removing things I didn't care about.

The SQL is pretty straightforward and just joins some tables, removes duplicates, and removes events and items we don't care about. The code to do the feature extraction is what you'd expect. Manual renaming, transformations, merging, and summarization. It's here for your amusement should you be so inclined:

{% highlight python %}
def clean_nl(nl_data_raw: pandas.DataFrame, curse_rating: pandas.DataFrame) -> pandas.DataFrame:
    nl_data_raw = nl_data_raw.sort_values(['video', 'run_number', 'floor_number'])
    curses = nl_data_raw[['item', 'video', 'run_number', 'floor_number']][nl_data_raw.event == 'Floor was Cursed']
    curses.rename(columns = {'item': 'curse'}, inplace = True)
    curses = pandas.merge(curses, curse_rating, how='inner', )

    wins = nl_data_raw[['video', 'run_number']][nl_data_raw.event == 'Won the run']
    wins['outcome'] = 1
    losses = nl_data_raw[['video', 'run_number']][nl_data_raw.event == 'Lost the run']
    losses['outcome'] = 0
    outcomes = pandas.concat((wins, losses))

    duration = nl_data_raw[['floor_duration', 'video', 'run_number', 'floor_number']].drop_duplicates()
    duration['floor_duration'] = duration.groupby(['video', 'run_number'])['floor_duration'].cumsum()

    items = nl_data_raw[['item', 'video', 'run_number', 'floor_number']][nl_data_raw.event == 'Item Collected']
    items = clean_items(item_ratings, items, item_match)

    character = nl_data_raw[['game_character', 'video', 'run_number', 'floor_number']].drop_duplicates()
    character = character.merge(characters, how='inner', left_on='game_character', right_on='nl_char')[['game_character', 'video', 'run_number', 'floor_number']]

    nl_data_clean = items.merge(curses, how='left', on=('video', 'run_number', 'floor_number'))
    nl_data_clean = nl_data_clean.merge(character, how='left', on=('video', 'run_number', 'floor_number'))
    nl_data_clean = nl_data_clean.merge(duration, how='left', on=('video', 'run_number', 'floor_number'))
    nl_data_clean = nl_data_clean.merge(outcomes, how='left', on=('video', 'run_number'))
    nl_data_clean['curse'] = nl_data_clean['curse'].fillna('No Curse')
    nl_data_clean['rating_y'] = nl_data_clean['rating_y'].fillna(0)
    nl_data_clean = nl_data_clean.dropna()
    nl_data_clean = nl_data_clean[nl_data_clean['floor_number'] <= 10]

    return nl_data_clean
{% endhighlight %}

Now as great as all this data is, if we really want to capture the state of a run, there are some crucial things missing:
- **Item Attributes** - Not all items are created equal and we'd like to have some notion of how good an item is when he picks it up
- **Stats** - As the game progresses your stats like damage, shot speed, health, etc. change and make a huge difference in your future success with the run.

The first one we can take care of pretty easily. I found a website, [isaacranks](https://www.isaacranks.com/afterbirthplus/ranks) that curates numerical ratings for each item in the game based on user votes. 

{% highlight python %}
def get_item_ratings() -> pandas.DataFrame:
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:76.0) Gecko/20100101 Firefox/76.0'
        }
    item_rank_json = requests.get('https://www.isaacranks.com/afterbirthplus/ranks', headers=headers).json()
    item_ranks_df = [(item['name'], item['rating']) for item in item_rank_json['items']]
    return pandas.DataFrame(item_ranks_df, columns = ('item', 'rating'))
{% endhighlight %}

This data source is great and pretty straightforward but some of the names don't match the names in the NL-db so I manually went through every item and mapped it. I could have done a fuzzy match or something but there's only like 300 items and most of them matched. This is great progress but Isaac is renowned for item *synergies* which means the unique ways in which 2 or more items interact. The NL data source captures this to some extent with transformations that occur after collecting every item in a set, but not the emergent ones that happen through normal play. It's very difficult to quantify this automatically so we will just stick with the per-item quality for now.

So we have all the easy stuff donw. Now to get at those stats we were talking about.

## Extracting Data From Videos
The only reason I attempted to extract the stats from the video is that the data is exceedingly well-formatted. The stats we're interested in are always in the same spot on the screen and in the same font and font color. The only thing that is hard is the varying background and the pseudo-handwritten, bold font style.  Let's get started. Here's an example frame from one of the runs:

![Frame Example](/assets/img/ex1610.png)

Most of the character stats are the numbers on the far left. Then we have health at the top left and the last 10 items he's picked up are on the right. There's actually a lot more information we could get from the frame but these are the easy ones that should be most predictive of run outcome. Now I wasn't kidding when I said the locations of these are hyper consistent. They're so consistent, we can just hardcode the locations to get all the images we care about.

{% highlight python %}
segments = {'space_item': (0, 0, 40, 40),
'coins': (20, 47, 37, 60),
'bombs': (20, 64, 37, 76),
'keys': (20, 80, 37, 92),
'speed': (20, 120, 58, 133),
'range': (20, 140, 58, 153),
'tears': (20, 161, 37, 173),
'shot_speed': (20, 182, 58, 193),
'damage': (20, 202, 58, 213),
'luck': (20, 222, 58, 233),  # infinity
'devil_chance': (20, 242, 58, 253),  # probably ruined
'angel_chance': (20, 262, 58, 273),
'trinket': (37, 331, 66, 357),
'hp1': (56, 7, 71, 19),
'hp2': (72, 7, 87, 19),
'hp3': (88, 7, 103, 19),
'hp4': (104, 7, 119, 19),
. . .
}
{% endhighlight %}

The awesome power of [opencv](https://docs.opencv.org/master/index.html) allows us to process all these little pieces really succinctly. For example, to determine what item was just picked up, I use [template matching](https://en.wikipedia.org/wiki/Template_matching) which cross-correlates one image with a candidate image and determines the pixel of the best match. So I downloaded images of every item in the game and created a single giant image and ran this matching function:

{% highlight python %}
def get_item(item: numpy.ndarray, all_items: numpy.ndarray) -> dict:
    w, h = item.shape[::-1]
    res = cv2.matchTemplate(all_items, item, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return (top_left, bottom_right)
{% endhighlight %}

which worked quite well on the examples I tested. I did manually scale the images and colored the background of the large image to be uniform but that's the only change I made. The health I determined by looking at the average color in the hardcoded bounding box and seeing if it's approximately red or blue. Pretty straightforward! The only downside to this is that template matching can be very expensive so scaling down the images may be necessary.

### Text-based Stats
Doing these simple image operations is easy and works for 90% of the stuff we want to do for this project. But a lot of the data we want are text on the screen. There's 2 approaches we could use for this. One is to use template matching again for each digit. This is probably the most accurate measure since the font and text color are always the same. But I wanted to try something else. So instead I used [pytesseract](https://pypi.org/project/pytesseract/) to do image to text extraction. This was pretty hit or miss. I tried a TON of different pre-processing steps like binarizing the text, dilating and eroding to get rid of small unwanted features, but the accuracy seemed to hover around 70%. Here's what I settled on eventually:

{% highlight python %}
ret, image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite(f'images/{name}.png', image)
text = pytesseract.image_to_string(image, config=r'--psm 7 --dpi 70 -c tessedit_char_whitelist=0123456789.')
print(f'{name}: {text}')
{% endhighlight %}

It seems to really struggle with sevens :/ but it was fun to get it kind of working and we can still fill in a lot of the points it missed by interpolating between values in post.

We can use these same simple techniques (template and color matching) to solve any of the data augmentation tasks I wanted to on this project. No CNNs or fancy transforms needed here. Even determining when he progresses to different floors and which character he chose is as simple as looking for specific colors or patterns in a frame. Once we collect all this data, we simply join it back to the original data set by video and floor and we've magically extracted quite a few additional features for predictions! And because this is extracted from the frame directly, we could theoretically perform real time predictions as he plays. Pretty neat I think.

## Conclusion
People often overlook image-based data sources because they think it's hard to extract what they want or they think they have to retrain ResNet or something. Hopefully this post showed that somethimes you can get a lot of value from simple techniques. Everything done here is quite fast and can easily be run on many videos. I didn't run it on a large set because this is just a hobby project and I didn't want to stress my internet and hard drive by downloading thousands of YouTube videos. Anyway, hope you enjoyed and will maybe try using images or videos to augment your next analysis project. 

Cheers,
Jeremy

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML" type="text/javascript"></script>