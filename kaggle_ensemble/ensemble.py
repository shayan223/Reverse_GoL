
'''######################     Validation Predictions     ####################'''

test_data = pd.read_csv('./data/test.csv')
#print(test_data)
test_data = test_data.drop('id', axis='columns')
test_deltas = test_data['delta']
test_data = test_data.drop('delta', axis='columns')
test_data = test_data.to_numpy()

#NOTE: only use below on CNN (to reshape)
test_data = np.reshape(test_data, reshaping_dim)

''' ###  Use if using encoder and not CNN  ###
#use encoder to convert end frame to start
encoded_solution = encoder.predict(test_data)
'''

###  Use for CNN  ###
encoded_solution = auto_encoder.predict(test_data)

#round all predictions to nearest whole number
encoded_solution = np.rint(encoded_solution)
#round negatives to zero to get GoL frame
encoded_solution[encoded_solution < 0] = 0

#print(encoded_solution)
#print(type(encoded_solution))
total_score = 0;
max_score = test_data.shape[0]

#for i in tqdm(range(encoded_solution.shape[0])):
for i in tqdm(range(1000)):
    if(checkStart(encoded_solution[i], test_data[i], test_deltas[i]+10,extra_dim=True)):
        total_score += 1

print('Accuracy: ', total_score/max_score)
