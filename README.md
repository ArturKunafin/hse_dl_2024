# Введение в глубокое обучение
# HSE SPRING 2024


- Большая часть материалов курса нагло позаимствована из других курсов.
- В репозитории к каждой неделе будут появлятся материалы.
- Материалы лежат в папках `/week*`.
- Если вы хотите скачать из репозитория конкретную папку, просто вставьте ссылку на неё [в сервис для скачки.](https://minhaskamal.github.io/DownGit/#/home), либо [разберитесь с git](https://githowto.com/ru) и сделайте pull. 

- 📦 Видео записи семинаров и лекций -> будут появляться в беседе курса после каждого занятия.
- 📦 Любые вопросы можно задать в чат курса, все важные объявлеия будут там же.

## Идеология курса 
Показать, что никакого искусственного интеллекта не существует.
Нейросетка — это просто ансамбль из регрессий.

## План курса

#### Feedforward NNs
01. - [MLP не отходя от браузера](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.57027&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
02. - [Бэкпроп на конкретном примере](http://www.habarov.spb.ru/lab_nnet/nn_js_lab/content/nn_js_backprop_2.html)
   - [Почему momentum работает](https://distill.pub/2017/momentum/)
   - [Методы оптимизации](https://habr.com/ru/post/318970/)
   - [Bias correction in Adam](https://www.youtube.com/watch?v=-0ZMU-gnm2g)
   - [Cyclical Learning Rate](https://medium.com/swlh/cyclical-learning-rates-the-ultimate-guide-for-setting-learning-rates-for-neural-networks-3104e906f0ae)   
03. - [BatchNorm explained](https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338)
   - [Custom weight decay](https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/)
   - [Inverted dropout](https://www.coursera.org/lecture/deep-neural-network/dropout-regularization-eM33A)
   - [Weight initialization](https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/weight_initialization_activation_functions/)
#### CV
04. - [Convolution in CNN explained](https://www.youtube.com/watch?v=KTB_OFoAQcc)
   - [Dilated convolutions](https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/)
05. - [ResNet](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)
   - [Метрики в задачах классификации](https://habr.com/ru/company/ods/blog/328372/)
   - [A Comprehensive Introduction to Different Types of Convolutions](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)
06. - [Что выучивает сеть?](https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b)
   - [Перенос стиля](https://towardsdatascience.com/neural-style-transfer-applications-data-augmentation-43d1dc1aeecc) 
   - [Denoising Autoencoders](https://medium.com/@garimanishad/reconstruct-corrupted-data-using-denoising-autoencoder-python-code-aeaff4b0958e)
   - [Автоэнкодеры](https://habr.com/ru/post/331382/)
07. - [StyleGAN3](https://nvlabs.github.io/stylegan3/)
   - [VAE ultimate guide & intuition](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
   - [GAN explained](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans/lecture/gIAJ0/putting-it-all-together)
   
#### NLP
08. - [MLE explained](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1)
   - [NLP guide от Lena Voita](https://lena-voita.github.io/nlp_course.html)
   - [Word2Vec tutorial с математикой и кодом](https://github.com/Yorko/mlcourse.ai/blob/main/jupyter_russian/tutorials/word2vec_demonzheg.ipynb)
   - [spaCy vs NLTK vs gensim](https://www.kaggle.com/faressayah/nlp-with-spacy-nltk-gensim)
09. - Stanford CS224N [all lectures](http://web.stanford.edu/class/cs224n/slides/). [Slides on RNN](http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture07-fancy-rnn.pdf)
   - [NLTK book](https://www.nltk.org/book/)
   - [Deep RNNs](https://www.coursera.org/lecture/nlp-sequence-models/deep-rnns-ehs0S)
10. - [BERT, ELMO explained](https://jalammar.github.io/illustrated-bert/)
   - [Transformers explained](https://jalammar.github.io/illustrated-transformer/)
   - [Why BERT is called bidirectional?](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73)
   - [Harvard transformer implementation](http://nlp.seas.harvard.edu/2018/04/03/attention.html)


## Домашки и проекты

Список домашек: 

1. Основы tensorflow и/или pytorch, сбор своих первых нейросетей на tensorflow и/или pytorch
2. Собираем нейросеть своими руками

### Книги:

- 📚 Николенко: «Глубокое обучение. Погружение в мир нейронных сетей.» — книга, которую всем рекомендую читать в первую очередь.
- 📚 Гудфеллоу, Бенджио: «Глубокое обучение.» — библия глубокого обучения с ответами на многие вопросы и хорошей математикой.
- 📚 Педро Домингос: «Верховный Алгоритм.» — книга для рефлексии и мыслях о том, когда же нас поработит искуственный интеллект.

### Онлайн-курсы:

- 🤖 [Advanced ML от Яндекса.](https://www.coursera.org/specializations/aml) Это для тех, кто хочет развиваться дальше. В специализации есть разные специфические курсы. Первый из них про нейронки. Код на Tensorflow. Версия библиотеки там пока что старая.
- 🤖 [Курс нейронок, который читают в ШАД и сколтехе.](https://github.com/yandexdataschool/Practical_DL/tree/master)  Есть варианты кода на разных фреймворках. Есть видео лекций на русском и английском.
- 🤖 Бесплатный [курс по tf от Google.](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187) Короткий. Покрывает весь базовый Keras. Все тетрадки выложены в colab. Внутри есть очень странные интервью.  
- 🤖 Бесплатный [курс по pytorch от Samsung.](https://stepik.org/course/50352/syllabus)  Состоит из двух частей: CV и NLP. Для тех, кто хочет писать на pytorch.
- 🤖 [Deep learning на пальцах.](https://dlcourse.ai) Семён из кремниевой долины записывает лекции на youtube. Объясняет и правда на пальцах. Причём сложные вещи. Круто смотреть его стримы в метро, пока куда-то едешь. 
