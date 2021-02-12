<img src="https://github.com/taishi-nammoto/content_based_recommended_engine/blob/main/Data/wordcloud.png" width="700">

[Click here](https://github.com/taishi-nammoto/content_based_recommended_engine/blob/main/content_based_recommended_engine.ipynb) to see how I created the image above and Content Based Recommended Engine.

# Content Based Recommended Engine

# Task Details:
With the help of this particular data set you have to build ***a recommended engine***. And your recommended engine will return maximum 10 movies name if an user search for a particular movie.

Recommended engine generally in three types <br>
***1.content Based recommended engine*** <br>
2.collaborative recommender engine <br>
3.hybrid recommended engine

# Goal Details:
Recommended engine must return 5 movie names and maximum it can return 10 movie names if an user search for a particular movie. This recommender engine should not give suggestion in between 1 to 4 and 6 to 10 it have to return 5 movie names for 10 movie names.

# Data source
https://www.kaggle.com/shivamb/netflix-shows

# Content Based Recommended engine

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

<tbody>
    <tr>
      <th>1324</th>
      <td>Chicken Kokkachi</td>
      <td>[comedies, movies, international]</td>
      <td>[marriage, man, learns, said, wake, supporting...</td>
      <td>1.000000</td>
      <td>0.735539</td>
      <td>1.735539</td>
    </tr>
    <tr>
      <th>885</th>
      <td>Bhaji In Problem</td>
      <td>[comedies, movies, international]</td>
      <td>[two, unaware, knows, man, arrives, life, old,...</td>
      <td>1.000000</td>
      <td>0.735079</td>
      <td>1.735079</td>
    </tr>
    <tr>
      <th>2155</th>
      <td>Fifty Year Old Teenager</td>
      <td>[comedies, movies, international]</td>
      <td>[life, younger, woman, married, act, falls, li...</td>
      <td>1.000000</td>
      <td>0.699812</td>
      <td>1.699812</td>
    </tr>
    <tr>
      <th>3460</th>
      <td>Kuch Kuch Hota Hai</td>
      <td>[comedies, movies, dramas, international]</td>
      <td>[girl, best, woman, wish, mother, sets, father...</td>
      <td>0.958004</td>
      <td>0.741664</td>
      <td>1.699668</td>
    </tr>
    <tr>
      <th>7532</th>
      <td>Well Done Abba</td>
      <td>[comedies, movies, international]</td>
      <td>[mired, bureaucracy, village, leave, husband, ...</td>
      <td>1.000000</td>
      <td>0.686940</td>
      <td>1.686940</td>
    </tr>
    <tr>
      <th>102</th>
      <td>3 Türken &amp; ein Baby</td>
      <td>[comedies, movies, international]</td>
      <td>[lives, exgirlfriend, three, care, dissatisfie...</td>
      <td>1.000000</td>
      <td>0.684217</td>
      <td>1.684217</td>
    </tr>
    <tr>
      <th>287</th>
      <td>About Time</td>
      <td>[comedies, movies, dramas, international]</td>
      <td>[travel, learns, lives, go, men, decides, woma...</td>
      <td>0.958004</td>
      <td>0.724101</td>
      <td>1.682105</td>
    </tr>
    <tr>
      <th>6942</th>
      <td>The Tribe</td>
      <td>[comedies, movies, international]</td>
      <td>[sensation, life, memory, mother, biological, ...</td>
      <td>1.000000</td>
      <td>0.680872</td>
      <td>1.680872</td>
    </tr>
    <tr>
      <th>4357</th>
      <td>My Step Dad: The Hippie</td>
      <td>[comedies, movies, international]</td>
      <td>[man, three, stop, take, married, widowed, adu...</td>
      <td>1.000000</td>
      <td>0.680013</td>
      <td>1.680013</td>
    </tr>
    <tr>
      <th>7393</th>
      <td>Varane Avashyamund</td>
      <td>[comedies, movies, dramas, international]</td>
      <td>[complex, marriage, lives, arranged, man, beco...</td>
      <td>0.958004</td>
      <td>0.718106</td>
      <td>1.676110</td>
    </tr>
  </tbody>

## License
[MIT](https://choosealicense.com/licenses/mit/)
