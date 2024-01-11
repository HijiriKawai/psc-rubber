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
matrix_0_to_40 = np.array([
                     [20.0, 0.0, 0.0],
                     [0.4, 19.6, 0.0],
                     [0.6, 0.3, 19.1],
                   ])
classes_0_to_40 = ["445g", "692g", "1118g"]
cm_0_to_40 = useful_graphs.ConfusionMatrix(matrix_0_to_40,class_list=classes_0_to_40)
cm_0_to_40.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_0_to_40_matrix_mean.png')


# 1500_to_3000の混合行列
# matrix_1500_to_3000 = np.array([
#                      [20.0, 0.0, 0.0],
#                      [6.7, 13.2, 0.1],
#                      [0.0, 0.0, 20.0],
#                    ])
# classes_1500_to_3000 = ["445g", "692g", "1118g"]
# cm_1500_to_3000 = useful_graphs.ConfusionMatrix(matrix_1500_to_3000,class_list=classes_1500_to_3000)
# cm_1500_to_3000.plot(to_normalize=True,to_show_number_label=True,text_font_size=10,save_path='result/iris/iris_1500_to_3000_matrix_mean.png')
