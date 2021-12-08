from DataBase import DataBase
import pandas as pd


class CRElectoralRoll(DataBase):

    cedColumn = "CEDULA"
    nameColumn = "NOMBRE"
    firstLastNameColumn = "1.APELLIDO"
    secondLastNameColumn = "2.APELLIDO"

    def __init__(self, directory="datasets", fileName="PADRON_COMPLETO.txt"):
        self.directory = directory
        self.fileName = fileName
        self.dataFrame = None

    def loadDatabase(self) -> None:
        """ This function loads the information contained in the database specified.
        """
        # Target columns to load
        columnsIdx = [0, 5, 6, 7]

        columnNames = [self.cedColumn, self.nameColumn,
                       self.firstLastNameColumn, self.secondLastNameColumn]

        # Defines the converter to be applied to the respective column
        converters = {self.nameColumn: str.strip,
                      self.firstLastNameColumn: str.strip,
                      self.secondLastNameColumn: str.strip}

        # Defines the database file path
        filePath = "{}\\{}".format(self.directory, self.fileName)

        # Loads the information in the dataframe
        self.dataFrame = pd.read_csv(filepath_or_buffer=filePath,
                                     sep=",",
                                     header=None,
                                     names=columnNames,
                                     usecols=columnsIdx,
                                     converters=converters,
                                     encoding='latin-1')

    def isAuthentic(self, numId: str, nameId: str) -> bool:
        """ This function takes an identification number and full name and,
            compares it to the one who is in the database to determine if the
            information is authentic.

        Args:
            numId (int): identification number.
            nameId (str): identification full name.

        Returns:
            bool: identification authenticity.
        """
        authentic = False

        if (not numId or not nameId):
            return authentic

        numId = int(numId)
        # Divides the full name into names and last names
        nameId = nameId.split()
        name = " ".join(nameId[:-2])
        firstLastName = nameId[-2]
        secondLastName = nameId[-1]

        # Filters the elements and takes all the ones that start with the
        # first number of the identification number
        df = self.dataFrame[self.dataFrame[self.cedColumn] // 10**8 ==
                            numId // 10**8]

        # If the identification number exists in the database
        if (numId in df[:][self.cedColumn].values):
            # Gets the element's index
            idx = df[df[self.cedColumn] == numId].index[0]
            # Gets the element's row
            person = df.loc[idx]
            # Compares the input attributes with those that exist in the database
            authentic = (person[self.nameColumn] == name)
            authentic &= (person[self.firstLastNameColumn] == firstLastName)
            authentic &= (person[self.secondLastNameColumn] == secondLastName)

        return authentic
