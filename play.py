import pandas as pd


class MyClass:
    def __init__(self):
        self._data = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]})

    def get_dataframe(self, name):
        # Perform some operations using the input argument to generate the DataFrame
        # For demonstration purposes, we'll simply return a subset of the original DataFrame
        return self._data[self._data['Name'] == name]

    @property
    def dataframe(self):
        return self.get_dataframe


# Create an instance of MyClass
obj = MyClass()

# Access the DataFrame using the property-like syntax with the input argument
df = obj.dataframe('Bob')

# Print the DataFrame
print(df)

print(obj.dataframe)