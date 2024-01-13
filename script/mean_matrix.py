import useful_graphs # 平均用の混合行列用
import matplotlib.pyplot as plt
import numpy as np
#%%

plt.figure(dpi=700)
# 0_to_200の混合行列
# matrix_0_to_200 = np.array([
#                      [20.0, 0.0, 0.0],
#                      [0.8, 19.2, 0.0],
#                      [1.8, 0.2, 18.0],
#                    ])
# classes_0_to_200 = ["445g", "692g", "1118g"]
# cm_0_to_200 = useful_graphs.ConfusionMatrix(matrix_0_to_200,class_list=classes_0_to_200)
# cm_0_to_200.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_200_matrix_mean.png')

# 0_to_40の混合行列
# matrix_0_to_40 = np.array([
#                      [20.0, 0.0, 0.0],
#                      [0.4, 19.6, 0.0],
#                      [0.6, 0.3, 19.1],
#                    ])
# classes_0_to_40 = ["445g", "692g", "1118g"]
# cm_0_to_40 = useful_graphs.ConfusionMatrix(matrix_0_to_40,class_list=classes_0_to_40)
# cm_0_to_40.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_40_matrix_mean.png')

# sa_matrix_0_to_40 = np.array([
#                      [19.0, 0.6, 0.4],
#                      [0.3, 19.5, 0.2],
#                      [0.0, 0.7, 19.3],
#                    ])
# classes_0_to_40 = ["445g", "692g", "1118g"]
# sa_cm_0_to_40 = useful_graphs.ConfusionMatrix(sa_matrix_0_to_40,class_list=classes_0_to_40)
# sa_cm_0_to_40.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/self-attention/sa_0_to_40_matrix_mean.png')

# matrix_0_to_80 = np.array([
#                      [19.8, 0.0, 0.2],
#                      [0.4, 18.9, 0.7],
#                      [0.0, 0.0, 20.0],
#                    ])
# classes_0_to_80 = ["445g", "692g", "1118g"]
# cm_0_to_80 = useful_graphs.ConfusionMatrix(matrix_0_to_80,class_list=classes_0_to_80)
# cm_0_to_80.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_80_matrix_mean.png')

# matrix_0_to_120 = np.array([
#                      [19.1, 0.6, 0.3],
#                      [0.4, 19.6, 0.3],
#                      [0.0, 0.0, 20.0],
#                    ])
# classes_0_to_120 = ["445g", "692g", "1118g"]
# cm_0_to_120 = useful_graphs.ConfusionMatrix(matrix_0_to_120,class_list=classes_0_to_120)
# cm_0_to_120.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_120_matrix_mean.png')

# matrix_0_to_160 = np.array([
#                      [20.0, 0.0, 0.0],
#                      [0.1, 19.9, 0.0],
#                      [0.1, 0.4, 19.5],
#                    ])
# classes_0_to_160 = ["445g", "692g", "1118g"]
# cm_0_to_160 = useful_graphs.ConfusionMatrix(matrix_0_to_160,class_list=classes_0_to_160)
# cm_0_to_160.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_160_matrix_mean.png')

# matrix_0_to_20 = np.array([
#                      [19.6, 0.4, 0.0],
#                      [0.0, 19.5, 0.5],
#                      [0.0, 0.0, 20.0],
#                    ])
# classes_0_to_20 = ["445g", "692g", "1118g"]
# cm_0_to_20 = useful_graphs.ConfusionMatrix(matrix_0_to_20,class_list=classes_0_to_20)
# cm_0_to_20.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_20_matrix_mean.png')

# matrix_0_to_10 = np.array([
#                      [19.8, 0.2, 0.0],
#                      [1.3, 17.7, 1.0],
#                      [0.8, 0.7, 18.5],
#                    ])
# classes_0_to_10 = ["445g", "692g", "1118g"]
# cm_0_to_10 = useful_graphs.ConfusionMatrix(matrix_0_to_10,class_list=classes_0_to_10)
# cm_0_to_10.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_10_matrix_mean.png')

# matrix_0_to_4 = np.array([
#                      [14.9, 2.9, 2.2],
#                      [2.9, 14.0, 3.1],
#                      [1.0, 4.4, 14.6],
#                    ])
# classes_0_to_4 = ["445g", "692g", "1118g"]
# cm_0_to_4 = useful_graphs.ConfusionMatrix(matrix_0_to_4,class_list=classes_0_to_4)
# cm_0_to_4.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_4_matrix_mean.png')

# matrix_40_to_80 = np.array([
#                      [19.8, 0.0, 0.2],
#                      [0.2, 19.8, 0.0],
#                      [0.7, 0.0, 19.3],
#                    ])
# classes_40_to_80 = ["445g", "692g", "1118g"]
# cm_40_to_80 = useful_graphs.ConfusionMatrix(matrix_40_to_80,class_list=classes_40_to_80)
# cm_40_to_80.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_40_to_80_matrix_mean.png')

# matrix_1500_to_1540 = np.array([
#                      [20.0, 0.0, 0.0],
#                      [3.9, 16.0, 0.1],
#                      [0.0, 0.1, 19.9],
#                    ])
# classes_1500_to_1540 = ["445g", "692g", "1118g"]
# cm_1500_to_1540 = useful_graphs.ConfusionMatrix(matrix_1500_to_1540,class_list=classes_1500_to_1540)
# cm_1500_to_1540.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_1500_to_1540_matrix_mean.png')

sa_matrix_1500_to_1540 = np.array([
                     [19.5, 0.4, 0.1],
                     [1.2, 18.1, 0.7],
                     [0.0, 1.1, 18.9],
                   ])
classes_1500_to_1540 = ["445g", "692g", "1118g"]
sa_cm_1500_to_1540 = useful_graphs.ConfusionMatrix(sa_matrix_1500_to_1540,class_list=classes_1500_to_1540)
sa_cm_1500_to_1540.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/self-attention/sa_1500_to_1540_matrix_mean.png')


# matrix_0_to_4000 = np.array([
#                      [19.9, 0.1, 0.0],
#                      [1.2, 18.5, 0.3],
#                      [0.0, 0.0, 20.0],
#                    ])
# classes_0_to_4000 = ["445g", "692g", "1118g"]
# cm_0_to_4000 = useful_graphs.ConfusionMatrix(matrix_0_to_4000,class_list=classes_0_to_4000)
# cm_0_to_4000.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_4000_matrix_mean.png')


# 1500_to_3000の混合行列
# matrix_1500_to_3000 = np.array([
#                      [20.0, 0.0, 0.0],
#                      [6.7, 13.2, 0.1],
#                      [0.0, 0.0, 20.0],
#                    ])
# classes_1500_to_3000 = ["445g", "692g", "1118g"]
# cm_1500_to_3000 = useful_graphs.ConfusionMatrix(matrix_1500_to_3000,class_list=classes_1500_to_3000)
# cm_1500_to_3000.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_1500_to_3000_matrix_mean.png')
