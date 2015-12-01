f = open('Data/best4linreg.csv', 'w')

for i in range(0, 10):
	for j in range(0,100):
		f.write("%d,%d,%d\n"%(i,j, i + 2 *j))


f.close()
