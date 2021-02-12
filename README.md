<img src="https://github.com/taishi-nammoto/content_based_recommended_engine/blob/main/Data/wordcloud.png" width="700">

[Click here](https://github.com/taishi-nammoto/content_based_recommended_engine/blob/main/content_based_recommended_engine.ipynb) to see how I created the image above and Content Based Recommended Engine.

# Content Based Recommended Engine

## Task Details:
With the help of this particular data set you have to build ***a recommended engine***. And your recommended engine will return maximum 10 movies name if an user search for a particular movie.

Recommended engine generally in three types <br>
***1.content Based recommended engine*** <br>
2.collaborative recommender engine <br>
3.hybrid recommended engine

## Goal Details:
Recommended engine must return 5 movie names and maximum it can return 10 movie names if an user search for a particular movie. This recommender engine should not give suggestion in between 1 to 4 and 6 to 10 it have to return 5 movie names for 10 movie names.

## Data source
https://www.kaggle.com/shivamb/netflix-shows

## Content Based Recommended engine

Step 1. Download pretrained Google word2vec model
~~~
import gensim.downloader as api

# Download pretrained Google word2vec model
path = api.load("word2vec-google-news-300", return_path=True)
~~~

Step 2. Load the trained model
~~~
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
~~~

Step 3. Find similarity of the selected movie and the other movies
~~~
description = model.n_similarity(df_clean['description'].iloc[i], df_selected['description'].iloc[0])
~~~

Step 4. Display a list of relevant movies

~~~
|      | title                                 |   total score |
|-----:|:--------------------------------------|--------------:|
| 5782 | Stand Up and Away! with Brian Regan   |       1.75374 |
| 6902 | The Standups                          |       1.72589 |
| 1471 | COMEDIANS of the world                |       1.71525 |
| 1470 | Comedians in Cars Getting Coffee      |       1.70354 |
| 6232 | The Comedy Lineup                     |       1.70025 |
| 7077 | Tiffany Haddish Presents: They Ready  |       1.69649 |
| 1600 | Daniel Sloss: Live Shows              |       1.69646 |
| 6175 | The Break with Michelle Wolf          |       1.69529 |
| 6517 | The Joel McHale Show with Joel McHale |       1.69193 |
| 2117 | Fary : Hexagone                       |       1.69154 |
~~~

## License
[MIT](https://choosealicense.com/licenses/mit/)
