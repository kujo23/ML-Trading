f = open('Data/best4KNN.csv', 'w')

for i in range(0, 10):
	for j in range(0,60):
		f.write("%d,%d,%d\n"%(i,j, i + 2 *j))
	for j in range(0,40):
		f.write("%d,%d,%d\n"%(i,j, -(i + 2 *j)))

f.close()
