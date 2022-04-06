from AAM import AA


#types={"Openness To Change": [0,10,5,14],"Self-Enhancement": [3,12,1,16],"Conservation": [4,13,6,15],"Self-Trancendence":[2,7,11,17]}

AAM = AA()

AAM.load_analysis(model_type = "CAA")
AAM.load_analysis(model_type = "OAA")

AAM.plot("CAA", plot_type="barplot",archetype_number=0)
AAM.plot("OAA", plot_type="barplot",archetype_number=0)