"""
Коля пришел в кинотеатр, и выбирает себе место, куда сесть.
Он выбирает только из одного ряда, и хочет сесть так,
чтобы между ним и ближайшим уже занятым местом в этом ряду было как можно большее расстояние.
На вход подается последовательность нулей и единиц, где 0 обозначает свободное место, а 1 - занятое.
Нужно вывести максимально возможное расстояние между Колей и ближайшим к нему уже занятым местом. Пример: [0, 0, 1, 1, 1]
"""
import math

# для каждой буквы запомним ее индекс на клаавиатуре
# теперь притбавим к ответу разницу между всеми соседними убквами