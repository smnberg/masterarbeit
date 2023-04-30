import os
import random
import tensorflow as tf
import numpy as np

def create_data_record(number_of_travelers: int):

    
    travelers = []
    health_records = []
    number_of_cities_visited = 30
    

    for k in range(number_of_travelers):
        possible_cities = [ 1, 2, 3, 4, 5]
        hidden_city_fan = random.choice(possible_cities)
        #create the rival city here which is just shifted by 3
        rival_city = (hidden_city_fan + 3) % 5 
        travel_record = []
        sick  = 0
        for j in range(number_of_cities_visited):
            possible_cities = [ 1, 2, 3, 4, 5]
            possible_cities.remove(hidden_city_fan)
            visited_city = random.choice(possible_cities)
            #event needs to visit his home needs to be rare enough so that guessing true doesn't work.
            rare_event = random.choices([True, False], cum_weights=[1/ number_of_cities_visited, 1])
            if([True] == rare_event):
                visited_city = hidden_city_fan

            if( j == number_of_cities_visited - 1):
                visited_city = rival_city

            heimspiel = random.choice([1, 0])
            #Ein selbstgestricktes one-hot-encoding fuer die 10 moglichen zustande die sich aus der kombination 5 cities und Heimspiel ja/nein ergeben 2*5 == 10
            index_record = (2*visited_city + heimspiel) % 10
            new_record = [0,0,0,0,0,0,0,0,0,0]
            new_record[index_record] = 1
            travel_record.append(new_record)
            
            if((hidden_city_fan == visited_city) and heimspiel):
                sick = 1
            
        travelers.append(travel_record)
        health_records.append([sick])

    
    return tf.convert_to_tensor(travelers), tf.convert_to_tensor(health_records)





if __name__ == "__main__":
    input, output = create_data_record(10)
    print(input)
    print(output)

