
import sys
print(sys.path)


from pmsampsize.pmsampsize import *

def test_pmsampsize():

    #Test 1
    expected = [623, 662, 221, 662]
    test = pmsampsize(type = "b", csrsquared = 0.288, parameters = 24, prevalence = 0.174)
    actual = [test["results_table"][0][1], test["results_table"][1][1], 
              test["results_table"][2][1], test["results_table"][4][1]]
    assert actual == expected


    #Test 2
    expected = [615, 660, 221, 660]
    test = pmsampsize(type = "b", cstatistic = 0.89, parameters = 24, prevalence = 0.174)
    actual = [test["results_table"][0][1], test["results_table"][1][1], 
              test["results_table"][2][1], test["results_table"][4][1]]
    assert actual == expected


    #Test 3
    expected = [5143, 1039, 5143, 5143]
    test = pmsampsize(type = "s", csrsquared = 0.051, parameters = 30, rate = 0.065, timepoint = 2, meanfup = 2.07)
    actual = [test["results_table"][0][1], test["results_table"][1][1], 
              test["results_table"][2][1], test["results_table"][4][1]]
    assert actual == expected   

    #Test 4
    expected = [918, 401, 259, 918, 918]
    test = pmsampsize(type = "c", rsquared = 0.2, parameters = 25, intercept = 1.9, sd = 0.6)
    actual = [test["results_table"][0][1], test["results_table"][1][1], 
              test["results_table"][2][1], test["results_table"][3][1], test["results_table"][5][1]]
    assert actual == expected        

