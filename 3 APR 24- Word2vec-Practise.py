import nltk
# lib for Word2vec

from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re


story= """Once upon a time in the enchanting land of Dreamtopia, there lived a beautiful and adventurous young girl named Barbie. With her sparkling eyes and radiant smile, Barbie brought joy to everyone she met.

Barbie loved to explore the wonders of Dreamtopia, a magical realm filled with colorful creatures and breathtaking landscapes. She would often embark on thrilling adventures with her best friends, spreading happiness wherever they went.

One day, while wandering through the enchanted forest, Barbie stumbled upon a mysterious rainbow waterfall. Intrigued by its shimmering waters, she decided to follow its path, unaware of the incredible journey that lay ahead.

As Barbie ventured deeper into the forest, she encountered fantastical creatures like talking unicorns, mischievous fairies, and wise old wizards. Each encounter filled her heart with wonder and awe, reaffirming her belief in the magic of Dreamtopia.

But Barbie's journey was not without challenges. Along the way, she faced treacherous obstacles and cunning villains who sought to thwart her quest. Yet, with courage and determination, Barbie overcame every obstacle, proving that kindness and bravery are the most powerful magic of all.

Finally, aft
er a series of trials and tribulations, Barbie reached the heart of Dreamtopia, where she discovered the true source of its magic – the power of friendship, love, and imagination. With a grateful heart, Barbie returned home, forever cherishing the memories of her extraordinary adventure in Dreamtopia.

And so, the tale of Barbie's magical journey became a legend in the land of Dreamtopia, inspiring generations to believe in the power of dreams and the magic that lies within us all."""


#text processing data

text = re.sub (r'\[[0-9]*\]',' ',story)
text = re.sub (r'\s+',' ',text) # removed extra lines and space
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s',' ',text)

# prepare dataset by tokenizing

from nltk.tokenize import sent_tokenize
sentences = nltk.sent_tokenize(text)

from nltk.tokenize import word_tokenize
sentences_word = [nltk.word_tokenize(sentence) for sentence in sentences]


for i in range(len(sentences)):
    sentences_word[i] = [word for word in sentences_word[i] if word not in stopwords.words('english')]
    
 # Training the Word2Vec model
#-------------------------------------------------------------
model = Word2Vec(sentences_word, min_count=1) 
model
#====================================================
#pip install gensim==3.8.3 # in this version only we have vocab
words = model.wv.index_to_key

#AttributeError: The vocab attribute was removed from KeyedVector in Gensim 4.0.0.
#Use KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.
#words = list(model.wv.key_to_index.keys())  # Get list of words in vocabulary


similar_barbie= model.wv.most_similar('barbie')
similar_friendship= model.wv.most_similar('friendship')
similar_obstacles= model.wv.most_similar('obstacles')
similar_forest= model.wv.most_similar('forest')


# still more research going on Word2vec.

