from AAM import AA

AAM = AA()
AAM.load_csv("ESS8_data.csv",range(12,33))
AAM.analyse()
AAM.plot()