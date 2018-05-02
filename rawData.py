import numpy as np

class RawData:
    def __init__(self):
        data1={'0.1': [[12.6, 43.100000000000001, 24.600000000000001, 44.5, 30.199999999999999, 19.100000000000001, 21.0, 198.0], [7.0999999999999996, 56.600000000000001, 198.69999999999999, 198.59999999999999, 198.69999999999999, 198.5, 198.40000000000001, 198.40000000000001], [3.3999999999999999, 198.69999999999999, 198.30000000000001, 198.5, 198.69999999999999, 198.30000000000001, 198.59999999999999, 198.5], [4.2999999999999998, 35.5, 199.69999999999999, 199.5, 199.40000000000001, 199.69999999999999, 199.59999999999999, 199.40000000000001], [7.7999999999999998, 35.100000000000001, 76.599999999999994, 112.3, 129.59999999999999, 148.59999999999999, 130.19999999999999, 148.30000000000001], [7.9000000000000004, 17.800000000000001, 18.300000000000001, 22.199999999999999, 11.6, 13.4, 15.800000000000001, 17.399999999999999], [10.699999999999999, 46.600000000000001, 66.299999999999997, 117.90000000000001, 96.700000000000003, 128.09999999999999, 110.59999999999999, 123.59999999999999], [6.2000000000000002, 199.5, 199.59999999999999, 199.69999999999999, 199.30000000000001, 199.19999999999999, 199.69999999999999, 199.40000000000001], [9.3000000000000007, 199.30000000000001, 199.09999999999999, 199.59999999999999, 199.30000000000001, 199.40000000000001, 199.59999999999999, 199.30000000000001], [6.5999999999999996, 198.0, 198.0, 198.0, 198.0, 198.0, 198.90000000000001, 198.59999999999999], [4.2000000000000002, 21.0, 198.0, 198.0, 198.90000000000001, 198.69999999999999, 198.90000000000001, 198.90000000000001], [8.8000000000000007, 178.5, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [6.4000000000000004, 199.19999999999999, 199.40000000000001, 199.30000000000001, 199.30000000000001, 199.09999999999999, 199.0, 199.19999999999999], [5.0999999999999996, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [3.7000000000000002, 8.3000000000000007, 17.399999999999999, 102.7, 80.099999999999994, 67.900000000000006, 77.700000000000003, 113.40000000000001], [10.300000000000001, 18.300000000000001, 179.5, 198.30000000000001, 198.19999999999999, 198.5, 179.0, 198.40000000000001], [8.9000000000000004, 72.400000000000006, 90.099999999999994, 89.099999999999994, 133.90000000000001, 104.40000000000001, 131.90000000000001, 170.30000000000001], [6.0, 165.09999999999999, 144.09999999999999, 123.2, 169.5, 185.09999999999999, 157.19999999999999, 164.40000000000001], [7.2999999999999998, 199.5, 199.69999999999999, 199.40000000000001, 199.5, 199.40000000000001, 199.59999999999999, 199.40000000000001], [10.199999999999999, 11.1, 198.0, 198.0, 198.40000000000001, 198.09999999999999, 198.19999999999999, 198.0]], '0.5': [[29.699999999999999, 199.69999999999999, 199.59999999999999, 199.69999999999999, 199.69999999999999, 199.80000000000001, 199.40000000000001, 199.40000000000001], [4.4000000000000004, 199.0, 199.0, 199.0, 199.19999999999999, 199.40000000000001, 199.5, 199.30000000000001], [8.3000000000000007, 199.69999999999999, 199.5, 199.69999999999999, 199.59999999999999, 199.59999999999999, 199.40000000000001, 199.40000000000001], [5.5, 198.0, 198.0, 87.900000000000006, 198.0, 198.0, 198.59999999999999, 198.80000000000001], [8.5, 31.399999999999999, 199.0, 199.0, 198.80000000000001, 199.30000000000001, 199.30000000000001, 199.5], [6.2000000000000002, 29.300000000000001, 37.700000000000003, 198.59999999999999, 198.80000000000001, 198.69999999999999, 199.59999999999999, 199.59999999999999], [3.7000000000000002, 74.900000000000006, 198.0, 198.0, 198.80000000000001, 199.40000000000001, 179.59999999999999, 199.40000000000001], [5.7999999999999998, 198.40000000000001, 198.40000000000001, 198.30000000000001, 198.30000000000001, 198.69999999999999, 198.59999999999999, 198.40000000000001], [5.7000000000000002, 10.300000000000001, 198.0, 198.0, 198.0, 198.80000000000001, 199.5, 199.80000000000001], [6.0999999999999996, 198.80000000000001, 199.19999999999999, 199.0, 199.09999999999999, 199.09999999999999, 198.80000000000001, 199.5], [10.1, 198.0, 198.0, 198.30000000000001, 199.19999999999999, 198.5, 199.30000000000001, 199.0], [11.9, 198.0, 199.5, 199.5, 199.40000000000001, 199.69999999999999, 199.40000000000001, 199.5], [9.4000000000000004, 119.7, 198.0, 198.80000000000001, 199.40000000000001, 199.59999999999999, 199.90000000000001, 199.5], [6.7000000000000002, 38.399999999999999, 199.19999999999999, 198.69999999999999, 198.40000000000001, 199.19999999999999, 199.40000000000001, 199.59999999999999], [9.0999999999999996, 199.30000000000001, 199.40000000000001, 199.40000000000001, 199.19999999999999, 199.40000000000001, 199.59999999999999, 199.69999999999999], [9.6999999999999993, 199.90000000000001, 199.59999999999999, 199.19999999999999, 199.40000000000001, 199.5, 199.69999999999999, 199.59999999999999], [4.0999999999999996, 199.40000000000001, 199.59999999999999, 199.80000000000001, 199.40000000000001, 199.69999999999999, 199.19999999999999, 199.40000000000001], [3.8999999999999999, 188.59999999999999, 198.0, 199.59999999999999, 199.19999999999999, 199.59999999999999, 199.5, 199.59999999999999], [10.6, 198.0, 199.40000000000001, 199.40000000000001, 199.09999999999999, 199.40000000000001, 199.5, 199.40000000000001], [9.1999999999999993, 38.299999999999997, 63.700000000000003, 198.09999999999999, 199.19999999999999, 199.19999999999999, 199.59999999999999, 199.59999999999999]], '0.01': [[7.0, 199.59999999999999, 199.5, 199.5, 199.40000000000001, 199.59999999999999, 199.59999999999999, 199.40000000000001], [5.2000000000000002, 7.0999999999999996, 15.6, 64.799999999999997, 45.799999999999997, 20.600000000000001, 49.0, 39.700000000000003], [6.5, 199.40000000000001, 199.30000000000001, 199.80000000000001, 199.5, 199.30000000000001, 199.40000000000001, 199.69999999999999], [7.2999999999999998, 62.799999999999997, 159.69999999999999, 120.8, 119.8, 121.5, 99.5, 198.40000000000001], [9.6999999999999993, 23.300000000000001, 75.900000000000006, 108.40000000000001, 103.0, 60.0, 63.899999999999999, 30.699999999999999], [12.6, 199.09999999999999, 199.09999999999999, 199.40000000000001, 199.19999999999999, 199.0, 199.09999999999999, 199.09999999999999], [8.3000000000000007, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0], [15.5, 45.200000000000003, 198.80000000000001, 198.80000000000001, 198.5, 199.09999999999999, 198.69999999999999, 198.69999999999999], [5.0999999999999996, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [6.2999999999999998, 198.40000000000001, 198.80000000000001, 198.69999999999999, 198.59999999999999, 198.40000000000001, 198.59999999999999, 198.69999999999999], [6.0999999999999996, 8.8000000000000007, 29.600000000000001, 59.600000000000001, 58.5, 69.0, 198.0, 198.0], [10.5, 28.399999999999999, 14.699999999999999, 44.5, 26.899999999999999, 27.600000000000001, 41.700000000000003, 37.700000000000003], [7.7999999999999998, 44.100000000000001, 67.299999999999997, 109.09999999999999, 177.5, 117.2, 123.2, 144.69999999999999], [13.1, 179.19999999999999, 198.0, 127.59999999999999, 162.0, 198.09999999999999, 198.0, 198.0], [8.6999999999999993, 141.09999999999999, 141.69999999999999, 159.80000000000001, 88.200000000000003, 86.200000000000003, 126.5, 125.09999999999999], [12.699999999999999, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [12.300000000000001, 58.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [5.2999999999999998, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [6.4000000000000004, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [5.2999999999999998, 41.0, 114.0, 198.5, 198.90000000000001, 199.0, 199.0, 198.59999999999999]]}
        data2={'0.01': [[11.4, 13.699999999999999, 29.199999999999999, 38.0, 70.400000000000006, 49.100000000000001, 47.0, 71.0], [4.4000000000000004, 82.900000000000006, 85.299999999999997, 104.7, 91.799999999999997, 84.5, 142.90000000000001, 85.200000000000003], [4.7000000000000002, 198.40000000000001, 199.5, 198.80000000000001, 198.69999999999999, 198.90000000000001, 198.40000000000001, 199.0], [25.699999999999999, 68.400000000000006, 138.40000000000001, 179.30000000000001, 168.0, 181.19999999999999, 119.2, 180.40000000000001], [7.2999999999999998, 62.799999999999997, 99.299999999999997, 198.0, 86.5, 92.799999999999997, 94.400000000000006, 135.40000000000001], [9.1999999999999993, 9.4000000000000004, 199.0, 198.90000000000001, 198.59999999999999, 198.80000000000001, 199.09999999999999, 198.59999999999999], [12.5, 162.69999999999999, 164.09999999999999, 142.80000000000001, 198.0, 198.0, 198.0, 198.0], [8.4000000000000004, 107.0, 144.0, 143.5, 146.5, 165.40000000000001, 199.0, 143.0], [9.0999999999999996, 199.19999999999999, 199.5, 199.5, 199.30000000000001, 199.40000000000001, 199.5, 199.5], [8.0999999999999996, 199.5, 199.40000000000001, 199.5, 199.5, 199.69999999999999, 199.30000000000001, 199.59999999999999], [7.2999999999999998, 22.0, 29.399999999999999, 198.59999999999999, 198.69999999999999, 198.69999999999999, 198.59999999999999, 198.30000000000001], [9.0, 53.200000000000003, 84.900000000000006, 107.90000000000001, 99.5, 114.2, 111.40000000000001, 79.799999999999997], [6.7999999999999998, 199.30000000000001, 199.5, 199.59999999999999, 199.80000000000001, 199.40000000000001, 199.59999999999999, 199.5], [5.7000000000000002, 47.799999999999997, 59.299999999999997, 49.299999999999997, 49.799999999999997, 61.600000000000001, 77.0, 77.400000000000006], [48.299999999999997, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [5.0999999999999996, 199.30000000000001, 198.69999999999999, 198.5, 198.59999999999999, 198.30000000000001, 198.5, 198.5], [10.699999999999999, 18.600000000000001, 120.40000000000001, 198.09999999999999, 198.0, 198.0, 198.09999999999999, 198.19999999999999], [6.2999999999999998, 159.69999999999999, 198.0, 179.30000000000001, 161.30000000000001, 159.40000000000001, 178.59999999999999, 198.0], [7.2999999999999998, 27.699999999999999, 33.100000000000001, 38.700000000000003, 98.900000000000006, 76.900000000000006, 91.700000000000003, 97.700000000000003], [7.7999999999999998, 181.30000000000001, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0]], '0.5': [[10.5, 198.0, 198.0, 198.69999999999999, 199.19999999999999, 199.5, 199.40000000000001, 199.09999999999999], [27.0, 198.0, 198.0, 198.80000000000001, 198.90000000000001, 198.40000000000001, 198.80000000000001, 198.80000000000001], [3.2999999999999998, 199.5, 199.5, 199.5, 199.90000000000001, 199.59999999999999, 199.59999999999999, 199.69999999999999], [4.5999999999999996, 199.69999999999999, 199.5, 199.30000000000001, 199.69999999999999, 199.90000000000001, 199.69999999999999, 199.40000000000001], [5.7999999999999998, 198.69999999999999, 198.30000000000001, 198.40000000000001, 198.59999999999999, 199.59999999999999, 199.69999999999999, 199.40000000000001], [7.7000000000000002, 199.30000000000001, 199.5, 199.59999999999999, 199.5, 199.19999999999999, 199.5, 199.40000000000001], [8.0, 35.700000000000003, 198.69999999999999, 198.5, 198.90000000000001, 199.30000000000001, 199.40000000000001, 199.40000000000001], [7.5999999999999996, 199.40000000000001, 199.5, 199.30000000000001, 199.40000000000001, 199.40000000000001, 199.5, 199.40000000000001], [8.8000000000000007, 53.0, 5.0999999999999996, 198.0, 198.0, 198.0, 198.0, 199.0], [6.0, 199.0, 199.0, 199.0, 199.0, 199.19999999999999, 199.19999999999999, 199.19999999999999], [13.699999999999999, 198.0, 198.0, 198.69999999999999, 198.59999999999999, 199.40000000000001, 199.59999999999999, 199.30000000000001], [7.5999999999999996, 198.0, 198.0, 198.0, 199.0, 199.0, 199.0, 199.69999999999999], [6.2000000000000002, 16.800000000000001, 117.90000000000001, 198.0, 198.40000000000001, 198.5, 199.69999999999999, 199.5], [9.0999999999999996, 198.0, 198.0, 198.0, 198.0, 198.0, 199.0, 199.0], [10.699999999999999, 100.7, 199.30000000000001, 199.19999999999999, 199.09999999999999, 199.09999999999999, 199.40000000000001, 199.59999999999999], [7.4000000000000004, 198.09999999999999, 198.30000000000001, 198.19999999999999, 198.5, 199.30000000000001, 199.5, 199.59999999999999], [7.2999999999999998, 199.59999999999999, 199.59999999999999, 199.40000000000001, 199.59999999999999, 199.69999999999999, 199.59999999999999, 199.59999999999999], [3.3999999999999999, 198.0, 198.59999999999999, 198.69999999999999, 199.0, 199.0, 199.0, 198.90000000000001], [9.0999999999999996, 198.5, 198.30000000000001, 198.5, 198.30000000000001, 179.59999999999999, 199.40000000000001, 199.69999999999999], [10.6, 198.80000000000001, 198.30000000000001, 198.69999999999999, 198.59999999999999, 199.30000000000001, 199.90000000000001, 199.40000000000001]], '0.1': [[6.4000000000000004, 18.199999999999999, 44.399999999999999, 55.299999999999997, 199.30000000000001, 199.30000000000001, 199.30000000000001, 199.59999999999999], [7.4000000000000004, 199.59999999999999, 199.59999999999999, 199.40000000000001, 199.59999999999999, 199.40000000000001, 199.59999999999999, 199.30000000000001], [8.5, 127.5, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [6.2999999999999998, 61.899999999999999, 199.40000000000001, 198.69999999999999, 199.30000000000001, 199.69999999999999, 199.40000000000001, 199.59999999999999], [3.6000000000000001, 108.0, 147.30000000000001, 119.59999999999999, 117.3, 124.0, 149.80000000000001, 164.30000000000001], [17.100000000000001, 198.59999999999999, 198.80000000000001, 198.59999999999999, 198.5, 198.59999999999999, 198.30000000000001, 198.40000000000001], [6.5999999999999996, 9.5, 9.1999999999999993, 10.1, 9.0999999999999996, 10.1, 7.0, 13.199999999999999], [2.2999999999999998, 24.800000000000001, 68.900000000000006, 46.799999999999997, 24.899999999999999, 198.59999999999999, 198.80000000000001, 198.69999999999999], [10.699999999999999, 14.1, 15.6, 19.199999999999999, 120.0, 103.3, 159.69999999999999, 198.0], [8.5, 198.90000000000001, 198.90000000000001, 199.0, 198.90000000000001, 198.69999999999999, 198.80000000000001, 198.90000000000001], [9.3000000000000007, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0], [4.5, 16.0, 91.700000000000003, 160.59999999999999, 122.5, 125.59999999999999, 125.59999999999999, 179.30000000000001], [7.7999999999999998, 199.40000000000001, 199.5, 199.19999999999999, 199.40000000000001, 199.59999999999999, 199.40000000000001, 199.40000000000001], [11.6, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [5.2999999999999998, 48.600000000000001, 130.59999999999999, 199.19999999999999, 199.19999999999999, 199.59999999999999, 199.09999999999999, 199.30000000000001], [4.4000000000000004, 199.40000000000001, 199.5, 199.5, 199.40000000000001, 199.40000000000001, 199.5, 199.40000000000001], [10.1, 198.0, 199.30000000000001, 198.59999999999999, 198.80000000000001, 198.90000000000001, 199.0, 198.5], [23.199999999999999, 53.299999999999997, 198.40000000000001, 198.59999999999999, 198.90000000000001, 198.59999999999999, 198.0, 198.90000000000001], [8.6999999999999993, 26.199999999999999, 163.0, 142.30000000000001, 105.0, 145.09999999999999, 181.40000000000001, 161.90000000000001], [7.7000000000000002, 19.300000000000001, 123.3, 132.69999999999999, 71.099999999999994, 195.09999999999999, 164.5, 119.09999999999999]]}
        data3={'0.5': [[4.2999999999999998, 74.599999999999994, 198.0, 198.0, 198.0, 198.40000000000001, 199.30000000000001, 199.59999999999999], [9.1999999999999993, 198.19999999999999, 198.19999999999999, 198.30000000000001, 198.59999999999999, 198.09999999999999, 199.0, 198.59999999999999], [10.699999999999999, 28.5, 199.09999999999999, 198.5, 198.5, 199.59999999999999, 199.5, 199.59999999999999], [4.9000000000000004, 23.800000000000001, 198.30000000000001, 199.09999999999999, 198.80000000000001, 199.40000000000001, 199.40000000000001, 199.40000000000001], [9.5, 29.199999999999999, 199.69999999999999, 199.30000000000001, 199.59999999999999, 199.40000000000001, 199.80000000000001, 199.80000000000001], [4.9000000000000004, 198.59999999999999, 198.80000000000001, 198.59999999999999, 198.5, 198.59999999999999, 199.59999999999999, 199.40000000000001], [4.7999999999999998, 199.5, 199.19999999999999, 198.90000000000001, 199.59999999999999, 199.69999999999999, 199.40000000000001, 199.5], [9.0, 19.100000000000001, 199.40000000000001, 199.90000000000001, 199.5, 199.40000000000001, 199.40000000000001, 199.59999999999999], [4.2999999999999998, 198.0, 198.0, 198.30000000000001, 198.5, 199.40000000000001, 199.80000000000001, 199.40000000000001], [7.5, 18.600000000000001, 199.40000000000001, 199.69999999999999, 199.5, 199.69999999999999, 199.40000000000001, 199.59999999999999], [7.0, 199.40000000000001, 199.30000000000001, 199.40000000000001, 199.59999999999999, 199.59999999999999, 199.40000000000001, 199.69999999999999], [8.5, 179.09999999999999, 198.0, 199.59999999999999, 199.40000000000001, 199.40000000000001, 199.40000000000001, 199.59999999999999], [5.5, 199.09999999999999, 199.0, 198.80000000000001, 199.59999999999999, 199.5, 199.40000000000001, 199.5], [12.4, 198.90000000000001, 199.09999999999999, 198.90000000000001, 199.5, 199.40000000000001, 199.09999999999999, 199.80000000000001], [3.8999999999999999, 199.0, 199.0, 199.19999999999999, 199.5, 199.40000000000001, 199.30000000000001, 199.5], [8.1999999999999993, 198.69999999999999, 198.30000000000001, 199.0, 199.0, 199.19999999999999, 199.5, 199.40000000000001], [4.2999999999999998, 198.19999999999999, 198.19999999999999, 198.40000000000001, 198.5, 199.40000000000001, 199.69999999999999, 199.80000000000001], [10.0, 199.80000000000001, 199.30000000000001, 199.5, 199.59999999999999, 199.5, 199.40000000000001, 199.30000000000001], [3.7999999999999998, 198.0, 198.0, 198.69999999999999, 199.0, 199.40000000000001, 199.69999999999999, 199.5], [9.9000000000000004, 11.5, 198.5, 198.69999999999999, 198.69999999999999, 199.19999999999999, 198.5, 198.40000000000001]], '0.01': [[5.2000000000000002, 198.5, 198.69999999999999, 198.69999999999999, 198.69999999999999, 198.80000000000001, 198.40000000000001, 198.19999999999999], [4.5, 199.69999999999999, 199.5, 199.30000000000001, 199.69999999999999, 199.40000000000001, 199.59999999999999, 199.40000000000001], [7.2000000000000002, 9.5, 44.700000000000003, 198.40000000000001, 198.80000000000001, 198.69999999999999, 198.5, 198.69999999999999], [5.2999999999999998, 199.40000000000001, 199.80000000000001, 199.30000000000001, 199.40000000000001, 199.30000000000001, 199.5, 199.5], [8.5, 38.899999999999999, 22.199999999999999, 43.700000000000003, 43.100000000000001, 30.100000000000001, 50.200000000000003, 38.600000000000001], [7.2999999999999998, 73.599999999999994, 37.899999999999999, 111.5, 159.80000000000001, 159.90000000000001, 160.40000000000001, 160.09999999999999], [4.0, 105.7, 180.90000000000001, 162.59999999999999, 165.40000000000001, 198.0, 198.0, 198.0], [5.7999999999999998, 198.69999999999999, 198.69999999999999, 198.80000000000001, 198.30000000000001, 198.80000000000001, 198.90000000000001, 198.30000000000001], [14.0, 17.300000000000001, 99.299999999999997, 75.299999999999997, 55.600000000000001, 73.599999999999994, 84.400000000000006, 46.200000000000003], [5.7999999999999998, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [64.5, 70.299999999999997, 84.799999999999997, 198.0, 198.19999999999999, 198.09999999999999, 198.0, 198.0], [8.6999999999999993, 102.8, 139.40000000000001, 140.5, 178.5, 139.59999999999999, 139.19999999999999, 178.30000000000001], [7.2000000000000002, 12.4, 54.600000000000001, 45.299999999999997, 39.899999999999999, 25.199999999999999, 11.699999999999999, 26.800000000000001], [4.5999999999999996, 198.40000000000001, 198.5, 198.19999999999999, 198.5, 198.09999999999999, 198.59999999999999, 198.19999999999999], [9.5999999999999996, 32.899999999999999, 58.5, 53.100000000000001, 158.5, 136.09999999999999, 76.5, 109.7], [6.2000000000000002, 198.5, 198.40000000000001, 198.59999999999999, 198.80000000000001, 198.09999999999999, 198.40000000000001, 198.09999999999999], [5.0, 198.5, 198.59999999999999, 198.59999999999999, 199.0, 198.80000000000001, 199.09999999999999, 198.69999999999999], [7.2999999999999998, 199.59999999999999, 199.30000000000001, 199.69999999999999, 199.40000000000001, 199.5, 199.30000000000001, 199.5], [8.6999999999999993, 87.200000000000003, 179.0, 159.0, 140.19999999999999, 159.69999999999999, 159.59999999999999, 161.90000000000001], [4.5, 199.30000000000001, 199.5, 199.59999999999999, 199.69999999999999, 199.5, 199.40000000000001, 199.59999999999999]], '0.1': [[5.7000000000000002, 8.8000000000000007, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0], [6.5, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0], [7.7000000000000002, 199.69999999999999, 199.5, 199.5, 199.59999999999999, 199.40000000000001, 199.5, 199.5], [5.7000000000000002, 45.0, 27.800000000000001, 84.200000000000003, 88.099999999999994, 158.80000000000001, 159.19999999999999, 159.90000000000001], [11.5, 41.299999999999997, 198.19999999999999, 198.09999999999999, 198.5, 198.09999999999999, 198.30000000000001, 198.59999999999999], [3.8999999999999999, 22.100000000000001, 43.399999999999999, 48.799999999999997, 25.800000000000001, 42.899999999999999, 50.299999999999997, 47.899999999999999], [6.7000000000000002, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [9.0999999999999996, 12.1, 21.199999999999999, 21.100000000000001, 39.299999999999997, 20.899999999999999, 28.399999999999999, 199.59999999999999], [7.2999999999999998, 180.59999999999999, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [25.300000000000001, 21.899999999999999, 178.90000000000001, 179.40000000000001, 178.80000000000001, 178.59999999999999, 198.0, 84.700000000000003], [6.7000000000000002, 63.700000000000003, 198.0, 198.0, 199.09999999999999, 199.0, 199.69999999999999, 199.59999999999999], [6.7000000000000002, 199.30000000000001, 199.30000000000001, 199.59999999999999, 199.90000000000001, 199.5, 199.69999999999999, 199.40000000000001], [10.199999999999999, 199.59999999999999, 199.59999999999999, 199.30000000000001, 199.69999999999999, 199.40000000000001, 199.59999999999999, 199.40000000000001], [9.0, 7.4000000000000004, 15.300000000000001, 15.199999999999999, 44.399999999999999, 38.200000000000003, 25.600000000000001, 23.300000000000001], [12.0, 53.0, 71.099999999999994, 149.09999999999999, 198.0, 198.0, 198.0, 198.0], [6.0999999999999996, 198.80000000000001, 199.0, 198.69999999999999, 198.80000000000001, 198.90000000000001, 198.80000000000001, 198.80000000000001], [4.4000000000000004, 2.1000000000000001, 11.800000000000001, 11.6, 90.700000000000003, 161.0, 178.69999999999999, 198.0], [45.399999999999999, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0], [9.8000000000000007, 199.0, 198.59999999999999, 198.80000000000001, 199.09999999999999, 199.0, 198.30000000000001, 198.90000000000001], [10.800000000000001, 12.6, 11.300000000000001, 11.699999999999999, 42.899999999999999, 34.600000000000001, 54.700000000000003, 42.700000000000003]]}
        data4={'0.01': [[4.0, 199.5, 199.30000000000001, 199.59999999999999, 199.5, 199.40000000000001, 199.69999999999999, 199.5], [25.0, 199.40000000000001, 199.19999999999999, 199.30000000000001, 199.19999999999999, 199.30000000000001, 199.30000000000001, 199.19999999999999], [5.7000000000000002, 199.5, 199.5, 199.40000000000001, 199.30000000000001, 199.69999999999999, 199.40000000000001, 199.59999999999999], [7.5, 9.5, 14.0, 22.600000000000001, 47.899999999999999, 198.59999999999999, 198.40000000000001, 198.69999999999999], [12.300000000000001, 79.599999999999994, 194.30000000000001, 146.09999999999999, 160.80000000000001, 167.0, 181.5, 151.59999999999999], [10.4, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [8.0999999999999996, 81.599999999999994, 199.09999999999999, 199.19999999999999, 199.09999999999999, 199.19999999999999, 199.19999999999999, 199.09999999999999], [7.0, 41.799999999999997, 44.5, 151.19999999999999, 162.0, 198.69999999999999, 198.90000000000001, 198.80000000000001], [6.7999999999999998, 199.59999999999999, 199.40000000000001, 199.40000000000001, 199.69999999999999, 199.40000000000001, 199.30000000000001, 199.19999999999999], [6.9000000000000004, 15.9, 198.59999999999999, 198.69999999999999, 198.30000000000001, 198.59999999999999, 198.90000000000001, 198.59999999999999], [6.2000000000000002, 199.19999999999999, 199.19999999999999, 199.19999999999999, 199.59999999999999, 199.09999999999999, 199.40000000000001, 199.30000000000001], [4.7999999999999998, 132.59999999999999, 161.59999999999999, 162.09999999999999, 179.19999999999999, 198.40000000000001, 161.90000000000001, 198.5], [24.600000000000001, 27.699999999999999, 41.399999999999999, 95.599999999999994, 57.5, 65.5, 36.799999999999997, 62.100000000000001], [5.2000000000000002, 25.399999999999999, 94.299999999999997, 148.09999999999999, 99.900000000000006, 137.30000000000001, 138.69999999999999, 85.200000000000003], [2.2000000000000002, 199.69999999999999, 199.69999999999999, 199.5, 199.80000000000001, 199.30000000000001, 199.5, 199.5], [7.0, 42.5, 198.40000000000001, 198.30000000000001, 198.30000000000001, 198.40000000000001, 198.30000000000001, 198.19999999999999], [9.1999999999999993, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0], [12.6, 29.300000000000001, 111.90000000000001, 142.80000000000001, 198.40000000000001, 198.19999999999999, 198.19999999999999, 198.40000000000001], [5.7000000000000002, 198.59999999999999, 198.5, 198.59999999999999, 198.40000000000001, 198.30000000000001, 199.09999999999999, 198.30000000000001], [10.9, 48.100000000000001, 198.90000000000001, 198.5, 198.59999999999999, 198.80000000000001, 198.80000000000001, 198.59999999999999]], '0.5': [[6.9000000000000004, 198.59999999999999, 198.5, 198.69999999999999, 199.40000000000001, 199.30000000000001, 199.59999999999999, 199.5], [6.4000000000000004, 199.09999999999999, 199.19999999999999, 199.5, 199.30000000000001, 199.40000000000001, 199.59999999999999, 199.69999999999999], [6.0999999999999996, 17.0, 198.5, 199.19999999999999, 199.09999999999999, 198.90000000000001, 199.30000000000001, 199.5], [6.7000000000000002, 199.0, 199.0, 199.0, 199.0, 199.0, 199.40000000000001, 199.69999999999999], [4.0, 16.600000000000001, 198.0, 198.0, 199.5, 199.19999999999999, 199.40000000000001, 199.5], [6.4000000000000004, 199.0, 199.0, 199.0, 199.5, 199.09999999999999, 199.59999999999999, 199.5], [15.0, 198.0, 198.80000000000001, 198.5, 199.40000000000001, 199.69999999999999, 199.59999999999999, 199.59999999999999], [12.0, 198.90000000000001, 199.40000000000001, 199.80000000000001, 199.40000000000001, 199.69999999999999, 199.69999999999999, 199.59999999999999], [2.8999999999999999, 83.900000000000006, 198.69999999999999, 198.5, 198.59999999999999, 198.80000000000001, 199.59999999999999, 199.69999999999999], [4.7999999999999998, 23.0, 198.0, 198.0, 198.59999999999999, 198.59999999999999, 198.40000000000001, 199.0], [6.2000000000000002, 199.59999999999999, 199.5, 199.5, 199.40000000000001, 199.19999999999999, 199.5, 199.5], [6.0, 199.5, 199.59999999999999, 199.69999999999999, 199.69999999999999, 199.69999999999999, 199.40000000000001, 199.5], [8.4000000000000004, 69.5, 199.0, 199.0, 199.40000000000001, 199.59999999999999, 199.5, 199.59999999999999], [7.5, 198.0, 198.0, 198.40000000000001, 199.69999999999999, 199.19999999999999, 199.59999999999999, 199.5], [11.4, 198.0, 198.0, 199.0, 198.80000000000001, 199.5, 199.30000000000001, 199.30000000000001], [6.2999999999999998, 199.59999999999999, 199.40000000000001, 199.80000000000001, 199.69999999999999, 199.5, 199.59999999999999, 199.5], [7.5, 161.80000000000001, 199.69999999999999, 199.5, 199.40000000000001, 199.5, 199.30000000000001, 199.40000000000001], [3.7999999999999998, 198.0, 198.09999999999999, 198.40000000000001, 198.69999999999999, 198.59999999999999, 199.69999999999999, 199.69999999999999], [8.6999999999999993, 159.0, 198.0, 199.30000000000001, 199.69999999999999, 199.5, 199.5, 199.59999999999999], [6.7000000000000002, 198.90000000000001, 199.5, 199.09999999999999, 199.09999999999999, 199.5, 199.59999999999999, 199.59999999999999]], '0.1': [[5.4000000000000004, 158.59999999999999, 178.5, 178.40000000000001, 198.0, 198.0, 198.0, 198.0], [9.8000000000000007, 52.899999999999999, 178.69999999999999, 149.5, 123.3, 102.0, 149.5, 122.5], [7.9000000000000004, 199.40000000000001, 199.69999999999999, 199.5, 199.5, 199.59999999999999, 199.80000000000001, 199.5], [7.9000000000000004, 9.0999999999999996, 10.800000000000001, 23.699999999999999, 92.799999999999997, 27.399999999999999, 44.100000000000001, 73.0], [11.800000000000001, 199.5, 199.40000000000001, 199.59999999999999, 199.80000000000001, 199.30000000000001, 199.5, 199.5], [5.7000000000000002, 15.9, 199.09999999999999, 199.09999999999999, 199.09999999999999, 199.19999999999999, 199.30000000000001, 199.30000000000001], [7.7000000000000002, 199.40000000000001, 199.5, 199.69999999999999, 199.59999999999999, 199.59999999999999, 199.30000000000001, 199.59999999999999], [9.4000000000000004, 179.59999999999999, 102.09999999999999, 179.90000000000001, 160.30000000000001, 199.0, 199.0, 199.0], [8.0, 16.300000000000001, 30.199999999999999, 44.600000000000001, 47.399999999999999, 48.600000000000001, 45.299999999999997, 60.600000000000001], [4.2999999999999998, 31.600000000000001, 24.899999999999999, 36.700000000000003, 33.700000000000003, 73.599999999999994, 48.200000000000003, 198.59999999999999], [8.8000000000000007, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [3.3999999999999999, 7.4000000000000004, 5.2999999999999998, 3.7999999999999998, 5.4000000000000004, 12.4, 12.199999999999999, 10.699999999999999], [7.7000000000000002, 61.200000000000003, 63.799999999999997, 167.09999999999999, 150.09999999999999, 116.90000000000001, 129.80000000000001, 122.3], [6.5999999999999996, 17.800000000000001, 11.300000000000001, 13.199999999999999, 16.800000000000001, 24.600000000000001, 32.200000000000003, 14.6], [7.9000000000000004, 198.19999999999999, 198.5, 198.69999999999999, 198.69999999999999, 198.80000000000001, 198.69999999999999, 198.90000000000001], [11.800000000000001, 14.199999999999999, 19.100000000000001, 34.399999999999999, 90.599999999999994, 52.799999999999997, 71.599999999999994, 88.599999999999994], [9.5, 16.899999999999999, 47.899999999999999, 17.899999999999999, 44.399999999999999, 38.700000000000003, 182.30000000000001, 180.5], [5.7999999999999998, 12.300000000000001, 141.5, 178.80000000000001, 144.0, 198.0, 198.0, 198.0], [9.1999999999999993, 53.899999999999999, 71.400000000000006, 78.400000000000006, 94.0, 72.299999999999997, 134.90000000000001, 72.700000000000003], [9.6999999999999993, 12.300000000000001, 10.699999999999999, 14.300000000000001, 17.399999999999999, 20.899999999999999, 18.100000000000001, 35.5]]}
        data5={'0.1': [[4.7000000000000002, 199.59999999999999, 199.5, 199.80000000000001, 199.40000000000001, 199.40000000000001, 199.69999999999999, 199.30000000000001], [7.5999999999999996, 198.0, 198.0, 198.0, 198.0, 198.0, 199.09999999999999, 198.90000000000001], [6.4000000000000004, 41.5, 43.299999999999997, 62.700000000000003, 55.799999999999997, 58.600000000000001, 49.899999999999999, 44.100000000000001], [6.7999999999999998, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [8.3000000000000007, 123.59999999999999, 72.599999999999994, 198.0, 198.0, 198.0, 198.0, 198.0], [7.2999999999999998, 9.0, 87.0, 89.400000000000006, 89.0, 198.0, 198.0, 198.0], [7.5, 39.200000000000003, 35.100000000000001, 24.600000000000001, 31.399999999999999, 199.30000000000001, 199.59999999999999, 199.5], [6.2000000000000002, 11.6, 20.0, 25.100000000000001, 21.699999999999999, 77.099999999999994, 50.600000000000001, 48.399999999999999], [15.0, 97.799999999999997, 199.40000000000001, 199.30000000000001, 199.30000000000001, 199.09999999999999, 199.09999999999999, 199.09999999999999], [8.5999999999999996, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0], [6.2000000000000002, 121.90000000000001, 84.299999999999997, 89.599999999999994, 74.400000000000006, 110.40000000000001, 198.0, 198.0], [5.5, 3.7999999999999998, 10.800000000000001, 15.6, 13.5, 17.199999999999999, 20.100000000000001, 34.5], [5.7000000000000002, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [4.5999999999999996, 128.80000000000001, 198.0, 198.40000000000001, 198.19999999999999, 198.09999999999999, 198.19999999999999, 198.09999999999999], [7.2000000000000002, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [9.1999999999999993, 124.7, 59.100000000000001, 83.599999999999994, 48.200000000000003, 77.400000000000006, 78.200000000000003, 199.09999999999999], [7.5, 24.199999999999999, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [5.2999999999999998, 47.700000000000003, 65.099999999999994, 98.5, 78.099999999999994, 61.5, 62.100000000000001, 115.2], [4.0999999999999996, 9.8000000000000007, 108.59999999999999, 127.2, 100.0, 113.7, 116.0, 93.299999999999997], [9.1999999999999993, 142.69999999999999, 179.59999999999999, 180.90000000000001, 160.09999999999999, 179.40000000000001, 198.0, 159.5]], '0.5': [[9.9000000000000004, 198.0, 198.0, 198.59999999999999, 198.30000000000001, 198.5, 199.59999999999999, 199.5], [6.4000000000000004, 199.59999999999999, 199.59999999999999, 199.19999999999999, 199.5, 199.59999999999999, 199.5, 198.90000000000001], [6.7999999999999998, 199.40000000000001, 199.69999999999999, 199.09999999999999, 199.59999999999999, 199.5, 199.5, 199.40000000000001], [8.8000000000000007, 13.300000000000001, 198.0, 198.0, 199.80000000000001, 199.59999999999999, 199.59999999999999, 199.30000000000001], [11.5, 199.59999999999999, 199.59999999999999, 199.40000000000001, 199.30000000000001, 199.40000000000001, 199.59999999999999, 199.19999999999999], [2.6000000000000001, 199.40000000000001, 199.5, 199.69999999999999, 199.5, 199.5, 199.59999999999999, 199.5], [5.0999999999999996, 20.600000000000001, 120.90000000000001, 198.0, 198.69999999999999, 198.80000000000001, 198.40000000000001, 199.5], [5.4000000000000004, 199.0, 199.0, 199.69999999999999, 199.40000000000001, 199.40000000000001, 199.5, 199.40000000000001], [22.5, 80.299999999999997, 198.59999999999999, 198.5, 199.0, 199.0, 199.0, 199.0], [6.7000000000000002, 31.800000000000001, 74.200000000000003, 198.59999999999999, 198.19999999999999, 199.0, 199.0, 199.0], [6.5999999999999996, 145.5, 179.0, 198.5, 199.19999999999999, 199.5, 199.59999999999999, 199.5], [7.0, 198.0, 198.0, 199.30000000000001, 199.69999999999999, 199.5, 199.5, 199.59999999999999], [13.6, 199.0, 199.0, 199.0, 199.59999999999999, 199.69999999999999, 199.09999999999999, 199.5], [8.5, 199.59999999999999, 199.59999999999999, 199.69999999999999, 199.30000000000001, 199.40000000000001, 199.19999999999999, 199.40000000000001], [6.4000000000000004, 31.100000000000001, 68.099999999999994, 199.69999999999999, 199.30000000000001, 199.30000000000001, 199.40000000000001, 199.30000000000001], [12.199999999999999, 34.0, 199.0, 199.0, 199.5, 199.69999999999999, 199.5, 199.69999999999999], [8.6999999999999993, 27.800000000000001, 198.5, 198.30000000000001, 198.5, 199.80000000000001, 199.30000000000001, 199.5], [11.6, 26.899999999999999, 80.599999999999994, 199.30000000000001, 199.30000000000001, 199.5, 199.40000000000001, 199.59999999999999], [5.5999999999999996, 199.40000000000001, 199.40000000000001, 199.69999999999999, 199.40000000000001, 199.59999999999999, 199.69999999999999, 199.69999999999999], [6.2999999999999998, 199.09999999999999, 199.69999999999999, 199.5, 199.5, 199.69999999999999, 199.30000000000001, 199.19999999999999]], '0.01': [[8.5, 29.699999999999999, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [5.4000000000000004, 199.59999999999999, 199.40000000000001, 199.5, 199.69999999999999, 199.80000000000001, 199.59999999999999, 199.59999999999999], [8.6999999999999993, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0, 198.0], [12.5, 121.09999999999999, 180.59999999999999, 161.69999999999999, 179.80000000000001, 199.0, 199.0, 199.0], [5.5999999999999996, 11.5, 117.90000000000001, 153.90000000000001, 131.90000000000001, 142.0, 136.0, 126.09999999999999], [8.5, 64.700000000000003, 198.09999999999999, 198.40000000000001, 198.09999999999999, 198.40000000000001, 198.0, 198.19999999999999], [9.3000000000000007, 74.299999999999997, 106.90000000000001, 47.799999999999997, 70.400000000000006, 89.799999999999997, 69.0, 104.09999999999999], [5.5, 23.899999999999999, 14.199999999999999, 11.6, 22.5, 20.199999999999999, 12.6, 13.300000000000001], [5.2000000000000002, 82.400000000000006, 93.299999999999997, 41.899999999999999, 100.90000000000001, 153.5, 83.299999999999997, 144.69999999999999], [11.800000000000001, 97.099999999999994, 198.5, 198.59999999999999, 199.30000000000001, 198.59999999999999, 198.80000000000001, 198.90000000000001], [5.5999999999999996, 14.4, 42.700000000000003, 37.299999999999997, 51.100000000000001, 48.799999999999997, 71.700000000000003, 36.799999999999997], [7.4000000000000004, 93.799999999999997, 198.5, 198.69999999999999, 198.5, 198.5, 198.5, 198.5], [11.0, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0, 199.0], [10.699999999999999, 122.40000000000001, 62.799999999999997, 198.0, 198.0, 198.0, 198.0, 198.0], [6.5999999999999996, 160.90000000000001, 142.30000000000001, 198.0, 198.0, 198.0, 198.0, 198.0], [8.9000000000000004, 149.30000000000001, 162.5, 131.90000000000001, 146.90000000000001, 144.80000000000001, 152.59999999999999, 164.30000000000001], [5.2999999999999998, 19.5, 30.399999999999999, 37.799999999999997, 85.099999999999994, 111.7, 34.799999999999997, 46.700000000000003], [4.2999999999999998, 199.19999999999999, 198.80000000000001, 198.5, 198.59999999999999, 199.0, 198.19999999999999, 198.59999999999999], [13.9, 29.399999999999999, 22.800000000000001, 22.300000000000001, 56.200000000000003, 68.700000000000003, 86.099999999999994, 40.0], [6.2999999999999998, 24.600000000000001, 29.100000000000001, 137.30000000000001, 156.69999999999999, 198.30000000000001, 144.40000000000001, 198.5]]}
        self.collatedData = [data1, data2, data3, data4, data5]
