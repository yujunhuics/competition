import matplotlib.pyplot as plt

file=open('f1score.txt')
# scores=file.read()
scores=file.readlines()
# print(file.readlines())
# print(len(scores))
# print(scores)
arr=[]
for i in scores:
    arr.append(i.strip())
print(arr)
y=[i for i in range(0,1927)]
print(y)
plt.plot(y,arr)
plt.show()