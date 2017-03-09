from ESN import ESN
from STESN import STESN
import numpy as np

from sklearn.utils import shuffle

from io import StringIO
from io import BytesIO

import matplotlib.pyplot as plt

tale = "once upon a time there lived in a certain village a little country girl, the prettiest creature who was ever seen. her mother was excessively fond of her and her grandmother doted on her still more. this good woman had a little red riding hood made for her. it suited the girl so extremely well that everybody called her little red riding hood. one day her mother, having made some cakes, said to her, go, my dear, and see how your grandmother is doing, for i hear she has been very ill. take her a cake, and this little pot of butter. little red riding hood set out immediately to go to her grandmother, who lived in another village. as she was going through the wood, she met with a wolf, who had a very great mind to eat her up, but he dared not, because of some woodcutters working nearby in the forest. he asked her where she was going. the poor child, who did not know that it was dangerous to stay and talk to a wolf, said to him, i am going to see my grandmother and carry her a cake and a little pot of butter from my mother. does she live far off said the wolf oh i say, answered little red riding hood it is beyond that mill you see there, at the first house in the village. well, said the wolf, and i will go and see her too. i will go this way and go you that, and we shall see who will be there first. the wolf ran as fast as he could, taking the shortest path, and the little girl took a roundabout way, entertaining herself by gathering nuts, running after butterflies, and gathering bouquets of little flowers. it was not long before the wolf arrived at the old woman is house. he knocked at the door tap, tap. who is there  your grandchild, little red riding hood, replied the wolf, counterfeiting her voice who has brought you a cake and a little pot of butter sent you by mother. the good grandmother, who was in bed, because she was somewhat ill, cried out, pull the bobbin, and the latch will go up. the wolf pulled the bobbin, and the door opened, and then he immediately fell upon the good woman and ate her up in a moment, for it been more than three days since he had eaten. he then shut the door and got into the grandmother is bed, expecting little red riding hood, who came some time afterwards and knocked at the door tap, tap. who is there  little red riding hood, hearing the big voice of the wolf, was at first afraid but believing her grandmother had a cold and was hoarse, answered, it is your grandchild little red riding hood, who has brought you a cake and a little pot of butter mother sends you. the wolf cried out to her, softening his voice as much as he could, pull the bobbin, and the latch will go up. little red riding hood pulled the bobbin, and the door opened. the wolf, seeing her come in, said to her, hiding himself under the bedclothes, put the cake and the little pot of butter upon the stool, and come get into bed with me. little red riding hood took off her clothes and got into bed. she was greatly amazed to see how her grandmother looked in her nightclothes, and said to her, grandmother, what big arms you have  all the better to hug you with, my dear. grandmother, what big legs you have  all the better to run with, my child. grandmother, what big ears you have  all the better to hear with, my child. grandmother, what big eyes you have  all the better to see with, my child. grandmother, what big teeth you have got  all the better to eat you up with. and, saying these words, this wicked wolf fell upon little red riding hood, and ate her all up. moral children, especially attractive, well bred young ladies, should never talk to strangers, for if they should do so, they may well provide dinner for a wolf. i say wolf, but there are various kinds of wolves. there are also those who are charming, quiet, polite, unassuming, complacent, and sweet, who pursue young women at home and in the streets. and unfortunately, it is these gentle wolves who are the most dangerous ones of all."
testText= "once upon a time in midwinter, when the snowflakes were falling like feathers from heaven, a queen sat sewing at her window, which had a frame of black ebony wood. as she sewed she looked up at the snow and pricked her finger with her needle. three drops of blood fell into the snow. the red on the white looked so beautiful that she thought to herself, if only i had a child as white as snow, as red as blood, and as black as the wood in this frame. soon afterward she had a little daughter who was as white as snow, as red as blood, and as black as ebony wood, and therefore they called her little snow white. and as soon as the child was born, the queen died. a year later the king took himself another wife. she was a beautiful woman, but she was proud and arrogant, and she could not stand it if anyone might surpass her in beauty. she had a magic mirror. every morning she stood before it, looked at herself, and said mirror, mirror, on the wall, who in this land is fairest of all to this the mirror answered you, my queen, are fairest of all. then she was satisfied, for she knew that the mirror spoke the truth. snow white grew up and became ever more beautiful. when she was seven years old she was as beautiful as the light of day, even more beautiful than the queen herself. one day when the queen asked her mirror mirror, mirror, on the wall, who in this land is fairest of all it answered you, my queen, are fair it is true. but snow white is a thousand times fairer than you. the queen took fright and turned yellow and green with envy. from that hour on whenever she looked at snow white her heart turned over inside her body, so great was her hatred for the girl. the envy and pride grew ever greater, like a weed in her heart, until she had no peace day and night. then she summoned a huntsman and said to him, take snow white out into the woods. i never want to see her again. kill her, and as proof that she is dead bring her lungs and her liver back to me. the huntsman obeyed and took snow white into the woods. he took out his hunting knife and was about to stab it into her innocent heart when she began to cry, saying, oh, dear huntsman, let me live. i will run into the wild woods and never come back. because she was so beautiful the huntsman took pity on her, and he said, run away, you poor child. he thought, the wild animals will soon devour you anyway, but still it was as if a stone had fallen from his heart, for he would not have to kill her. just then a young boar came running by. he killed it, cut out its lungs and liver, and took them back to the queen as proof of snow whites death. the cook had to boil them with salt, and the wicked woman ate them, supposing that she had eaten snow whites lungs and liver. the poor child was now all alone in the great forest, and she was so afraid that she just looked at all the leaves on the trees and did not know what to do. then she began to run. she ran over sharp stones and through thorns, and wild animals jumped at her, but they did her no harm. she ran as far as her feet could carry her, and just as evening was about to fall she saw a little house and went inside in order to rest. inside the house everything was small, but so neat and clean that no one could say otherwise. there was a little table with a white tablecloth and seven little plates, and each plate had a spoon, and there were seven knives and forks and seven mugs as well. against the wall there were seven little beds, all standing in a row and covered with snow white sheets. because she was so hungry and thirsty snow white ate a few vegetables and a little bread from each little plate, and from each mug she drank a drop of wine. afterward, because she was so tired, she lay down on a bed, but none of them felt right one was too long, the other too short until finally the seventh one was just right. she remained lying in it, entrusted herself to god, and fell asleep. after dark the masters of the house returned home. they were the seven dwarfs who picked and dug for ore in the mountains. they lit their seven candles, and as soon as it was light in their house they saw that someone had been there, for not everything was in the same order as they had left it. the first one said, who has been sitting in my chair the second one, who has been eating from my plate the third one, who has been eating my bread the fourth one, who has been eating my vegetables the fifth one, who has been sticking with my fork the sixth one, who has been cutting with my knife the seventh one, who has been drinking from my mug"
encodingString = "abcdefghijklmnopqrstuvwxyz ,."

tale_encoded = np.array(list(map(lambda c : encodingString.index(c), tale)))
test_encoded = np.array(list(map(lambda c : encodingString.index(c), testText)))

x_train = np.zeros((tale_encoded.shape[0], 29))
for i in range(x_train.shape[0]):
    x_train[i,tale_encoded[i]] = 1.0

x_test = np.zeros((test_encoded.shape[0], 29))
for i in range(x_test.shape[0]):
    x_test[i,test_encoded[i]] = 1.0

print(x_train)

x_train = np.vstack((x_train[:100,:], x_train))
x_test = np.vstack((x_test[:100,:], x_test))
print(x_train.shape)

esn = ESN(n_input=29, n_output=29, n_reservoir=500, random_seed=42, noise_level=0.001, leak_rate=0.7, spectral_radius=0.8, sparity=1.0)
esn.fit(inputData=x_test, outputData=x_test, transient_quota=0.025)

#for i in range(10):

testSeq = "she live far off said the wolf oh i say, answered "

def reencode(encoded):
    res = ""
    for i in range(encoded.shape[0]):
        res += encodingString[np.argmax(encoded[i])]
    return res

def upa(y):
    max_ind = np.argmax(y[:,0])
    y[:,0] = 0
    y[max_ind,0] = 1.0

predRes = esn.predict(x_test[101:,:], initial_input=x_test[100,:], continuation=False, initial_data=x_test[0:100,:])
predRes = predRes.T
print(predRes)

predResEnc = np.argmax(predRes,axis=1)

diff = np.count_nonzero(predResEnc != test_encoded[1:])
err = np.sum(np.abs(diff))
print(err)

predResString = reencode(predRes)

#Y = esn.generate(n=15000, continuation=True, initial_input=x_train[-1,:])

print(predResString)
print(reencode(x_test[101:,:]))
