# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 23:57:49 2022
@author: marcos
"""
# Python Tips
# https://www.youtube.com/@Indently
# %%
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

# %%
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a * b
d = np.dot(a, b)

x = {1, 2, 3, 4, 5}

x.add(5)
x.add(6)

number = 3
print(f"The NUMBER is {number}")
# %%


print(math.pow(2, 10))
print([x * 2 for x in range(1, 10)])
# %%
characters = ["I", "S", "C"]
actors = ["I2", "S2", "C2"]


print({x: y for x in characters for y in actors})


ilala = [x for x in range(20)]
# %% ZIP Function
a = ("John", "Charles", "Mike")
b = ("Jenny", "Christy", "Monica")

x = zip(a, b)

# use the tuple() function to display a readable version of the result:

print(tuple(x))
# %%
a = 1_000_000
b = 3_555

print(f"{a:, }")

# %%
# List to Dictionary
a = ["pamonha", "cactus", "sabonete"]
b = [1, 20, 4]

dicionario = dict(zip(a, b))

# %%
# F-Strings
name = "mauricio"
fstr = f"{name} é bobo"
f_str = f"{2+2}"
r_str = r"C:\Users\Documents"  # evita backslash hell

fr_str = rf"C:\Users\Documents\{2+2}"
# %%
# Extend - junta as listas
list_a = [1, 2, 3]
list_b = [5, 7, 9]

# Cria uma nova variavel
list_c = list_a + list_b
# sem criar uma nova variavel
list_a.extend(list_b)

# %%
# Avoid IF-ELSE Hell
# instead:
var = 0
if var == 1:
    print("1")
if var == 2:
    print("2...")

# Jeito certo: Usar um dicionario


def first():
    print("1")


def second():
    print("2")


def third():
    print("3")


def default():
    print("Padrão")


var = 0
funcs: dict = {0: first, 1: second, 2: third}


finals = funcs.get(var, default)
finals()

# %%
# Swap variables
a = 1
b = 2

# instead c=a, a=b, b=c
# use:
a, b = b, a

# %% ########### EPIC #############
text = " EPIC "
print(f"{text:@<20}")
print(f"{text:@>20}")
print(f"{text:.^30}")
print(f"  {text:#^20}")

# %% ENUMERATE
# When you use enumerate(),  the function gives you back two loop variables:
values = ["a", "b", "c", "d"]
for count, value in enumerate(values, start=1):
    print(count, value)


# %% GENERATE FOR LOOPS
process = [1, 2, 3]
asd = ["a", "b", "c"]
results = [item * 2 for item in asd]
print(results)

# %% MAP function


def addition(n):
    return n + n


# map() can listify the list of strings individually

test = list(map(list, asd))
print(test)
test = list(map(addition, process))
print(test)
test = list(map(lambda x, y: x + y, [1, 3, 5], [2, 0, -2]))
print(test)

# %% Describe function


def description():

    """
    Parameters
    -------------
    a = sadasdas

    b = asdfgasfd

    Returns
    -------------
    abc = int
    """

    print("nada")


# %% Treat Errors


class DivisionBy0(ZeroDivisionError):
    ...


denominator = 0

try:
    b = 2 / denominator
    if denominator == 0:
        raise DivisionBy0
except Exception as e:
    e
    b = 0

print(b)

# %% Apply lambda to columns

data = {'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Vehicles': [10, 20, 15, 25, 18, 22]}
df = pd.DataFrame(data)

df.loc[:, 'month'] = df['month'].apply(
                                    lambda x:
                                    "Jan1"
                                    if x == 'Jan'
                                    else x
                                )

print(df)

# %% Jointplot

# Create a sample DataFrame
data = {'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Vehicles': [10, 20, 15, 25, 18, 22]}
df = pd.DataFrame(data)

# Create a joint plot
sns.jointplot(x='month', y='Vehicles', data=df)

# Show the plot
plt.show()
# %%
# We map each temporal variable onto a circle such that the lowest value for that variable
# appears right next to the largest value.
# We compute the x- and y- component of that point using sin and cos trigonometric functions.

# Create a sample DataFrame
data = {'day': [1, 5, 10, 15, 20],
        'month': [1, 2, 3, 4, 5],
        'year': [2022, 2022, 2022, 2022, 2022]}
df = pd.DataFrame(data)

# Print unique values of 'month', 'day', and 'year'
print('Unique values of month:', df.month.unique())
print('Unique values of day:', df.day.unique())
print('Unique values of year:', df.year.unique())

# Calculate sine and cosine transformations for 'day' and 'month'
df['day_sin'] = np.sin(df.day * (2. * np.pi / 31))
df['day_cos'] = np.cos(df.day * (2. * np.pi / 31))
df['month_sin'] = np.sin((df.month - 1) * (2. * np.pi / 12))
df['month_cos'] = np.cos((df.month - 1) * (2. * np.pi / 12))

# Print the updated DataFrame
print(df)

# %%
# Data description
print("Rows     : ", df.shape[0])
print("Columns  : ", df.shape[1])
print("\nFeatures : \n", df.columns.tolist())
print("\nMissing values :  ", df.isnull().sum().values.sum())
print("\nUnique values :  \n", df.nunique())

# %%
# Facet Grid
# Load a sample dataset
tips = sns.load_dataset("tips")

# Create a FacetGrid
grid = sns.FacetGrid(tips, col="time", row="smoker")

# Map a plot type to the FacetGrid
grid.map(sns.scatterplot, "total_bill", "tip")

# Set the titles for each subplot
grid.set_titles("{row_name} {col_name}")

# Adjust the spacing between subplots
plt.subplots_adjust(top=0.9, hspace=0.3)

# Show the plot
plt.show()
# %%
# Pairplot
# Load a sample dataset
iris = sns.load_dataset("iris")

# Create a pair plot
sns.pairplot(iris, hue="species")

# Show the plot
plt.show()

# %%
# Cross Table
# Create a sample DataFrame
data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Age': [25, 30, 35, 40, 45],
    'Region': ['North', 'South', 'North', 'North', 'South'],
    'Income': [5000, 6000, 7000, 8000, 9000]
}
df = pd.DataFrame(data)

# Create a cross-tabulation table
cross_tab = pd.crosstab(df['Gender'], df['Region'])

# Print the cross-tabulation table
print(cross_tab)
