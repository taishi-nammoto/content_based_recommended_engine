# Content Based Recommended Engine
[This is the script](https://github.com/taishi-nammoto/content_based_recommended_engine/blob/main/content_based_recommended_engine.ipynb) 

<img src="https://github.com/taishi-nammoto/content_based_recommended_engine/blob/main/Data/wordcloud.png" width="700">

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
~~~
"outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>listed_in</th>\n",
       "      <th>description</th>\n",
       "      <th>score: listed_in</th>\n",
       "      <th>score: description</th>\n",
       "      <th>total score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1324</th>\n",
       "      <td>Chicken Kokkachi</td>\n",
       "      <td>[comedies, movies, international]</td>\n",
       "      <td>[marriage, man, learns, said, wake, supporting...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.735539</td>\n",
       "      <td>1.735539</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>885</th>\n",
       "      <td>Bhaji In Problem</td>\n",
       "      <td>[comedies, movies, international]</td>\n",
       "      <td>[two, unaware, knows, man, arrives, life, old,...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.735079</td>\n",
       "      <td>1.735079</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2155</th>\n",
       "      <td>Fifty Year Old Teenager</td>\n",
       "      <td>[comedies, movies, international]</td>\n",
       "      <td>[life, younger, woman, married, act, falls, li...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.699812</td>\n",
       "      <td>1.699812</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3460</th>\n",
       "      <td>Kuch Kuch Hota Hai</td>\n",
       "      <td>[comedies, movies, dramas, international]</td>\n",
       "      <td>[girl, best, woman, wish, mother, sets, father...</td>\n",
       "      <td>0.958004</td>\n",
       "      <td>0.741664</td>\n",
       "      <td>1.699668</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7532</th>\n",
       "      <td>Well Done Abba</td>\n",
       "      <td>[comedies, movies, international]</td>\n",
       "      <td>[mired, bureaucracy, village, leave, husband, ...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.686940</td>\n",
       "      <td>1.686940</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>102</th>\n",
       "      <td>3 Türken &amp; ein Baby</td>\n",
       "      <td>[comedies, movies, international]</td>\n",
       "      <td>[lives, exgirlfriend, three, care, dissatisfie...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.684217</td>\n",
       "      <td>1.684217</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>287</th>\n",
       "      <td>About Time</td>\n",
       "      <td>[comedies, movies, dramas, international]</td>\n",
       "      <td>[travel, learns, lives, go, men, decides, woma...</td>\n",
       "      <td>0.958004</td>\n",
       "      <td>0.724101</td>\n",
       "      <td>1.682105</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6942</th>\n",
       "      <td>The Tribe</td>\n",
       "      <td>[comedies, movies, international]</td>\n",
       "      <td>[sensation, life, memory, mother, biological, ...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.680872</td>\n",
       "      <td>1.680872</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4357</th>\n",
       "      <td>My Step Dad: The Hippie</td>\n",
       "      <td>[comedies, movies, international]</td>\n",
       "      <td>[man, three, stop, take, married, widowed, adu...</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.680013</td>\n",
       "      <td>1.680013</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7393</th>\n",
       "      <td>Varane Avashyamund</td>\n",
       "      <td>[comedies, movies, dramas, international]</td>\n",
       "      <td>[complex, marriage, lives, arranged, man, beco...</td>\n",
       "      <td>0.958004</td>\n",
       "      <td>0.718106</td>\n",
       "      <td>1.676110</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        title                                  listed_in  \\\n",
       "1324         Chicken Kokkachi          [comedies, movies, international]   \n",
       "885          Bhaji In Problem          [comedies, movies, international]   \n",
       "2155  Fifty Year Old Teenager          [comedies, movies, international]   \n",
       "3460       Kuch Kuch Hota Hai  [comedies, movies, dramas, international]   \n",
       "7532           Well Done Abba          [comedies, movies, international]   \n",
       "102       3 Türken & ein Baby          [comedies, movies, international]   \n",
       "287                About Time  [comedies, movies, dramas, international]   \n",
       "6942                The Tribe          [comedies, movies, international]   \n",
       "4357  My Step Dad: The Hippie          [comedies, movies, international]   \n",
       "7393       Varane Avashyamund  [comedies, movies, dramas, international]   \n",
       "\n",
       "                                            description  score: listed_in  \\\n",
       "1324  [marriage, man, learns, said, wake, supporting...          1.000000   \n",
       "885   [two, unaware, knows, man, arrives, life, old,...          1.000000   \n",
       "2155  [life, younger, woman, married, act, falls, li...          1.000000   \n",
       "3460  [girl, best, woman, wish, mother, sets, father...          0.958004   \n",
       "7532  [mired, bureaucracy, village, leave, husband, ...          1.000000   \n",
       "102   [lives, exgirlfriend, three, care, dissatisfie...          1.000000   \n",
       "287   [travel, learns, lives, go, men, decides, woma...          0.958004   \n",
       "6942  [sensation, life, memory, mother, biological, ...          1.000000   \n",
       "4357  [man, three, stop, take, married, widowed, adu...          1.000000   \n",
       "7393  [complex, marriage, lives, arranged, man, beco...          0.958004   \n",
       "\n",
       "      score: description  total score  \n",
       "1324            0.735539     1.735539  \n",
       "885             0.735079     1.735079  \n",
       "2155            0.699812     1.699812  \n",
       "3460            0.741664     1.699668  \n",
       "7532            0.686940     1.686940  \n",
       "102             0.684217     1.684217  \n",
       "287             0.724101     1.682105  \n",
       "6942            0.680872     1.680872  \n",
       "4357            0.680013     1.680013  \n",
       "7393            0.718106     1.676110  "
      ]
~~~

## License
[MIT](https://choosealicense.com/licenses/mit/)
