---
layout: default
title:  "Recommending Anime with Neural Networks"
date:   2020-12-13 12:55:30 -0500
categories: computer_vision
---

## Introduction
I was recently interviewing and one of the questions was about recommendation systems. I was familiar with collaberative filtering and its variants but had never looked into fancier ways of doing it. I found this [dataset](https://www.kaggle.com/CooperUnion/anime-recommendations-database) on Kaggle and some corresponding notebooks showing how to use Neural Networks for recommendation, but none used Keras to my satisfaction and I wanted to try some other approaches. So here's my go at creating an Anime Recommendation Engine using neural network embeddings.

## The Data
### Cleaning
The data consists of 2 files. The first, anime.csv contains metadata about all the animes in the database. The second, ratings.csv contains all the user-anime-rating triples. There's a bit of cleaning to do here but nothing too crazy. Mainly we're just cutting down the size of the data set to the top 1000 users and top 500 animes just to save on compute time. Then we re-index the users and animes, drop duplicate values, smash the 2 tables together, and scale the rating with a MinMaxScaler. Note I did some EDA before all this to figure out what I needed to do but I didn't think it was that interesting. 

{% highlight python %}
anime = pandas.read_csv('anime/anime.csv')
ratings = pandas.read_csv('anime/rating.csv')

user_votes = ratings.groupby('user_id').count().reset_index().sort_values('rating', ascending=False)
good_users = user_votes.iloc[:1000]
anime_votes = ratings.groupby('anime_id').count().reset_index().sort_values('rating', ascending=False)
good_animes = anime_votes.iloc[:500]

anime = anime[anime.anime_id.isin(good_animes.anime_id)]
ratings = ratings[ratings.user_id.isin(good_users.user_id)]
ratings = ratings[ratings.anime_id.isin(good_animes.anime_id)]
anime = anime.reset_index()
anime['anime_index'] = anime.index
anime.drop('index', axis=1, inplace=True)
anime.rename(columns={'rating': 'avg_rating'}, inplace=True)
users = ratings['user_id'].drop_duplicates().reset_index()
users['user_index'] = users.index
ratings = ratings \
    .merge(users[['user_id', 'user_index']], on='user_id') \
    .merge(anime[['anime_id', 'anime_index']], on='anime_id')
ratings = ratings[['rating', 'user_index', 'anime_index']]
ratings['rating_scaled'] = MinMaxScaler().fit_transform(ratings.rating.values.reshape(-1,1))
{% endhighlight %}

### Features
This is where most of my play time went. There are a couple things I tried with extracting additional features from the data. The most interesting was encoding the genres of the animes. The genre column has a comma seperated list of applicable genres like so: "Action, Adventure, Shounen, Super Power". I took the liberty of assuming these were in descending order of relevance based on my spot check of ones I'd seen. So I created a custom feature vector for the genres based on their position in the list. I dubbed this a "weighted one-hot encoding". I effectively wrote some code to one-hot encode every genre, but weight the position of the vector where the first genre appears higher than the second, the second higher than the third, etc. That looks like this:

{% highlight python %}
genre_encoder = OneHotEncoder(sparse=False)

def weighted_one_hot(anime_df: pandas.DataFrame):
    genres = anime_df.genre.str.split(', ', expand=True)
    unique_genres = pandas.unique(genres.values.ravel('K')).reshape(-1,1)
    genre_encoder.fit(unique_genres)
    weighted_vectors = []
    for anime_row in genres.values:
        importance = genres.shape[1]
        vector = numpy.zeros(unique_genres.size)
        for genre in anime_row:
            if genre:
                vector += importance * genre_encoder.transform(numpy.array([genre]).reshape(-1, 1))[0]
                importance -=1
            else:
                break
        weighted_vectors.append(vector)
    return numpy.array(weighted_vectors)
{% endhighlight %}

Then I played with PCA-ing the weighted one-hot matrix of genre features. I found that reducing it to 10 components tended to work out well. I could have of course created an embedding layer for the genres as well but to be honest I didn't feel like it. Now let's look at some recommendation approaches.

## Recommendation
### Normie Collaberative Filtering
There are a hundred sites explaining what collaverative filtering is so I'm not going to. Suffice to say we just need to create a user-item matrix and either do some dimensionality reduction on it or measure the distance between the user vectors as-is in that super high-dimensional space. If you choose the latter, cosine distance is recommended since the user vectors will be sparse and you don't need to take the 0 elements of the vector into account when computing cosine distance. Conceptually. The only difference between cosine distance and your normal *l2* distance is that cosine distance measures the angle between 2 vectors whereas *l2* measures the straight line distance. I chose to do dimensionality reduction via [Truncated SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) so I could get pretty pictures like this:

![User Space](/assets/img/svd_reco.png)

This is a 2D representation of all our users based on the animes they like. If we single out those 2 little dots at the top, we can see that their ratings are indeed quite similar:

![Similar Peeps](/assets/img/user_sim.png)

Neat! From there we just need to aggregate the highest-rated animes of similar users and recommend those to the user in question. This is easy on a small scale where you can iterate through each one, but on a large scale, you'd want to use an efficient nearest-neighbors algorithm like [Locality-Sensitive Hashing](https://frakt0x90.github.io/algorithms/2020/11/29/making-graphs-lsh.html)

### Embeddings
All the cool kids these days use neural networks for everything so we should too... Right? The idea behind it is pretty straightforward if you conceptually understand how embeddings work. We create an embeding layer for users and a second for the animes, we then conatenate those 2 embedding vectors with any other metadata we have about the users and animes and pass that feature vector through some hidden layers and try to predict the rating of a given user-anime pair. Let's see this in action:

{% highlight python %}
hidden_units = (32, 8)
user_embedding_size = 16
anime_embedding_size = 16

user_id_input = keras.Input(shape=(1,), name='user_id')
anime_id_input = keras.Input(shape=(1,), name='anime_id')
user_embedded = Embedding(ratings.user_index.max() + 1, user_embedding_size, input_length=1, name='user_embedding')(user_id_input)
anime_embedded = Embedding(ratings.anime_index.max() + 1, anime_embedding_size, input_length=1, name='anime_embedding')(anime_id_input)
concatenated = Concatenate()([user_embedded, anime_embedded])
out = Flatten()(concatenated)

for hidden in hidden_units:
    out = Dense(hidden, activation='relu')(out)

out = Dense(1, activation='linear', name='prediction')(out)

model = keras.Model(inputs=[user_id_input, anime_id_input], outputs=out)
model.summary()

model.compile('adam', loss='MSE', metrics=['MAE'])
{% endhighlight %}

If you're not familiar with the Keras functional API, and the use of the Model() class rather than the more often-seen Sequential() class, now is a good time to get acquainted. The functional API treats a neural network like a digraph of nested function calls. Just like [functional programming](https://en.wikipedia.org/wiki/Purely_functional_programming) languages like Haskell. This gives a nice structure and lets us create comlicated computational graphs in a simple interface. The gist of it is that that variable in parentheses at the end of a layer is the data we're passing to that layer. Let's train and see what happens:

{% highlight python %}
model.compile('adam', loss='MSE', metrics=['MAE'])

history = model.fit(
    [ratings.user_index, ratings.anime_index],
    ratings.rating_scaled,
    epochs=10,
    validation_split=.1
    )
{% endhighlight %}

Note here that we're not using any of the anime metadata we worked so hard to get. This is just anime and user embeddings sent to 2 hidden layers. At least on my machine, this results in a test loss of about .03 and starts overfitting pretty quickly. Let's add in that metadata and see what happens:

{% highlight python %}
ratings_nn = ratings.merge(anime_nn, on='anime_index')
scaled_columns = ratings_nn.columns[4:]
ratings_nn[scaled_columns] = MinMaxScaler().fit_transform(ratings_nn[scaled_columns])
anime_columns = ratings_nn.drop(['user_index', 'anime_index', 'rating', 'rating_scaled'], axis=1).shape[1]

hidden_units = (16, 8)
user_embedding_size = 8
anime_embedding_size = 8

user_id_input = keras.Input(shape=(1,), name='user_id')
anime_id_input = keras.Input(shape=(1,), name='anime_id')
anime_input = keras.Input(shape=(anime_columns,), name='anime')
anime_reshaped = Reshape((1, anime_columns))(anime_input)  # Input vector is (None, 8) but after embedding, those are (None, 1, embedding_size) so we reshape to get them to fit together
user_embedded = Embedding(ratings_nn.user_index.max() + 1, user_embedding_size, input_length=1, name='user_embedding')(user_id_input)
anime_embedded = Embedding(ratings_nn.anime_index.max() + 1, anime_embedding_size, input_length=1, name='anime_embedding')(anime_id_input)
concatenated = Concatenate()([user_embedded, anime_embedded, anime_reshaped])
out = Flatten()(concatenated)

for hidden in hidden_units:
    out = Dense(hidden, activation='relu')(out)

out = Dense(1, activation='linear', name='prediction')(out)

model = keras.Model(inputs=[user_id_input, anime_id_input, anime_input], outputs=out)
model.summary()
{% endhighlight %}

So we're just adding in one more input layer of the anime metadata and sticking that to the end of our concatenated embedding vectors. The Reshape() layer is simply to make the 2D anime features into a 3D tensor to fit on the end of the embedding output. Just so we're clear, I know I'm not doing my due-diligence with proper train/test/validation splitting but it's just a demo so forgive me. 

This model is definitely doing better. I'm getting validation loss of about .02. Let's see what kind of recommendations it's giving. First let's look at the anime embedding vectors to see which animes it thinks are similar.

![Similar Animes](/assets/img/sim_animes.png)

I told asked it what the most similar animes are to Dragon Ball Z since I watched that religiously as a child. We can see that the genres are pretty similar across the recommendations which is a good sign. I had to look up all of these and at least in the case of Bleach Movie, the clips I saw seemed DBZ-esque with cheesy villain speeches and big arena fights. 

Finally we can pick a user and get recommendations for them. In this case I just picked user 1. The predictions are 

![User Recs](/assets/img/user_recs.png)

Again the genres are pretty consistent which is promising and they are all highly rated. I would say that's not bad for this simple attempt. If we look at the distribution of redicted ratings for every anime for user 1, we get this 

![Rating Distribution](/assets/img/rating_dist.png)

Which shows there are some animes we are pretty sure they'd hate, a few we think they'd love and quite a few in the middle. This distribution would be interesting to explore later. 

## Conclusion
Creating a simple embedding model for recommendations isn't that hard and provides a very flexible framework to customize how the recommendations are computed. We could make this really fancy by adding in LSTM layers that could encode time-based preferences. For example if they're binging a series or are really into slice of life shows this month, we'd want to catch that pattern. Neural Networks allow us this flexibility. I will say that while I was researching this, there was a paper questioning the superiority of a neural network recommendation system of simple collaberative filtering (How dare they!). I can't say whether or not they're correct but it's good to be aware that the literature doesn't just whole-heartedly support these embedding models. Basic collaberative filtering still works REALLY well and should probably be your first try when doing something like this. Especially since there are packages that let you incorporate metadata into the basic approach without going full neural network on it. Anyway, hope you enjoyed and can try some of these approaches on your next recommendation project. 

Cheers,
Jeremy