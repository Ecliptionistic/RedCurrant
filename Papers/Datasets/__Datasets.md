# Наборы данных для распознавания эмоций по речи (SER)

| Набор данных | Год  | Состав                                                                              | Эмоции                                                                                                                     | Размер                                                    |    Язык    |
| ------------ | :--: | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | :--------: |
| **IEMOCAP**  | 2008 | 10 актеров (5М / 5Ж), диалоги (импровизация и сценарий); аудио, видео, mocap, текст | 9 категорий (нейтральный, гнев, радость, грусть, отвращение, страх, удивление, фрустрация, возбуждение)                    | ~12 часов (~10 039 реплик)                                | Английский |
| **EmoDB**    | 2005 | 10 профессиональных актеров (5М / 5Ж), чтение 10 стандартных фраз; аудио            | 7 категорий: гнев, скука, отвращение, страх, радость, грусть, нейтральный                                                  | 535 аудиозаписей (~25 минут)                              |  Немецкий  |
| **CREMA-D**  | 2014 | 91 актер (48М / 43Ж) различного этнического происхождения; аудио и видео            | 6 категорий (гнев, отвращение, страх, радость, нейтральный, грусть) с 4 уровнями интенсивности                             | 7 442 клипа (~2.45 часа)                                  | Английский |
| **RAVDESS**  | 2018 | 24 профессиональных актера (12М / 12Ж), речь и пение; аудио и видео                 | 8 категорий для речи: нейтральный, спокойный, радость, грусть, гнев, страх, отвращение, удивление (2 уровня интенсивности) | 1 440 аудиозаписей речи (всего 2 452 с песней, ~3.5 часа) | Английский |
| **DUSHA**    | 2021 | Краудсорсинг + подкасты (сотни дикторов); аудио и текстовые транскрипции            | 5 категорий: нейтральный (neutral), гнев (angry), радость (positive), грусть (sad), прочее (other)                         | ~350+ часов (~300 000+ реплик)                            |  Русский   |

@article{IEMOCAP,
  title     = {{IEMOCAP}: Interactive emotional dyadic motion capture database},
  author    = {Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
  journal   = {Language Resources and Evaluation},
  volume    = {42},
  number    = {4},
  pages     = {335--359},
  year      = {2008},
  publisher = {Springer}
}

@inproceedings{EmoDB,
  title     = {A database of German emotional speech},
  author    = {Burkhardt, Felix and Paeschke, Astrid and Rolfes, Miriam and Sendlmeier, Walter F and Weiss, Benjamin},
  booktitle = {Proceedings of Interspeech},
  volume    = {5},
  pages     = {1517--1520},
  year      = {2005}
}

@article{CREMA-D,
  title     = {{CREMA-D}: Crowd-sourced emotional multimodal actors dataset},
  author    = {Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Ruben C and Nenkova, Ani and Verma, Ragini},
  journal   = {IEEE Transactions on Affective Computing},
  volume    = {5},
  number    = {4},
  pages     = {377--390},
  year      = {2014},
  publisher = {IEEE}
}

@article{RAVDESS,
  title     = {The {Ryerson} Audio-Visual Database of Emotional Speech and Song ({RAVDESS}): A dynamic, multimodal set of facial and vocal expressions in {North American English}},
  author    = {Livingstone, Steven R and Russo, Frank A},
  journal   = {PLOS ONE},
  volume    = {13},
  number    = {5},
  pages     = {e0196391},
  year      = {2018},
  publisher = {Public Library of Science}
}

@inproceedings{DUSHA,
  title     = {{DUSHA}: Russian dataset for speech emotion recognition},
  author    = {Smirnov, Alexander and Andrusenko, Anton and Bakhteev, Oleg and Kavalerov, Nikita and Belskaya, Maria and Potafeev, Alexander},
  booktitle = {arXiv preprint arXiv:2111.07767},
  year      = {2021}
}
