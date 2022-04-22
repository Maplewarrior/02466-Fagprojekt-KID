from AAM import AA

AAM = AA()
# AAM.load_csv("ESS8_data.csv",range(12,33), rows = 1000)
# AAM.analyse(AA_type = "OAA")
# AAM.plot(model_type="OAA")

AAM.create_synthetic_data(N = 20000, M=10,K=3)
AAM.analyse(AA_type = "CAA", with_synthetic_data=True,K=3)
AAM.plot(model_type = "CAA", plot_type = "mixture_plot", with_synthetic_data=True)
