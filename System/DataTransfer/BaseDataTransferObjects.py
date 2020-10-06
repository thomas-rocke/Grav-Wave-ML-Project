class BaseDataInput():
    def __init__(self, inputDict):
        self._keys = inputDict.keys()
        for key, val in inputDict.items():
            self.__setattr__(key, val) #Passes each item in the dictionary to a separate attribute, which is accessed as self.key

    def ProcessData(self):
        #Perform any data manipulation, and populate self._inputs
        #Defaults to just populating self._inputs
        #Called before data is sent to the neural net
        self._inputs = [self.__getattribute__(key) for key in self._keys]

class BaseResultData():
    def __init__(self, resultDict):
        self._keys = resultDict.keys()
        for key, val in resultDict.items():
            self.__setattr__(key, val) #Passes each item in the dictionary to a separate attribute, which is accessed as self.key

    def DataToNet(self):
        #Defines all processing to convert result data before it is sent to the neural net.
        #Defaults to just populating self._results
        #Called before result data is sent to the neural net
        self._results = [self.__getattribute__(key) for key in self._keys]

    def NetToData(self):
        #Defines space for any processing to turn Neural Net outputs into useful data
        #Called before the predicted result data is sent to analysis functions
        pass


