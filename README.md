<img src="https://github.com/taishi-nammoto/content_based_recommended_engine/blob/main/Data/wordcloud.png" width="800">

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
Target item: 
|      | title      | listed_in                                                |
|-----:|:-----------|:---------------------------------------------------------|
| 5219 | Rishta.com | ['shows', 'international', 'comedies', 'tv', 'romantic'] |

Recommendation: 
|      | title                |   total score |
|-----:|:---------------------|--------------:|
| 3676 | Little Things        |      0.806124 |
| 7225 | Trio and a Bed       |      0.800453 |
| 3741 | Love & Anarchy       |      0.796251 |
| 4133 | Miss Culinary        |      0.795564 |
| 4293 | Murphy's Law of Love |      0.793992 |
| 7743 | Yours Fatefully      |      0.789017 |
| 4143 | Miss Rose            |      0.782283 |
| 3750 | Love Cheque Charge   |      0.781745 |
| 3252 | Just You             |      0.780499 |
| 2785 | Home for Christmas   |      0.779805 |
~~~

## License
[MIT](https://choosealicense.com/licenses/mit/)
