import os
import random

class KGBuilder(object):
    def __init__(
        self,
        out_train="data/train.txt",
        out_test="data/test.txt",
        out_kg="data/kg.txt",
        test_ratio=0.2):
        
        self.out_train = out_train
        self.out_test = out_test
        self.out_kg = out_kg
        self.test_ratio = test_ratio
        self.user_age_group = {}
        self.course_topic = {}
        self.likes = []
        self.kg_triples = []
    
    def _map_age_to_group(self, age):
        age = int(age)
        if age == 1:
            return 'child'
        elif age in [18, 25]:
            return 'teen'
        else:
            return 'adult'
    
    def load_users(self):
        # with open(self.users_file, 'r') as f:
        #     for line in f:
        #         parts = line.strip().split("::")
        #         if len(parts) < 5:
        #             continue
        #         user_id, gender, age, occ, zip_code = parts
        #         age_group = self._map_age_to_group(int(age))
        #         self.user_age_group[user_id] = age_group
        #         self.kg_triples.append(("user_" + user_id, "has_age", "agegroup_" + age_group))

        # user_id -> is enrolled in -> degree_program 
        pass
    
    def load_movies(self):
        # with open(self.courses_file, 'r') as f:
        #     for line in f:
        #         parts = line.strip().split("::")
        #         if len(parts) < 3:
        #             continue
        #         course_id, title, genres = parts
        #         genre_list = [genre.lower() for genre in genres.split("|")]
        #         self.course_topic[course_id] = genre_list
        #         for genre in genre_list:
        #             self.kg_triples.append(("course_" + course_id, "is_genre", "genre_" + genre.replace(" ", "_")))

        # course -> is part of -> degree_program
        pass

    def something(self):

        # course -> is of topic -> topic
        pass

    def load_ratings(self, min_rating=4):
        # with open(self.ratings_file, 'r') as f:
        #     for line in f:
        #         parts = line.strip().split("::")
        #         if len(parts) < 4:
        #             continue
        #         user_id, course_id, rating, timestamp = parts
        #         if int(rating) >= min_rating:
        #             self.likes.append(("user_" + user_id, "likes", "course_" + course_id))

        # user_id -> likes -> topics  
        pass  
    
    def split_sets(self):
        random.shuffle(self.likes)
        test_size = int(len(self.likes) * self.test_ratio)
        test_likes = self.likes[:test_size]
        train_likes = self.likes[test_size:]
        
        with open(self.out_train, 'w') as f_train:
            for triple in train_likes:
                f_train.write("%s\t%s\t%s\n" % triple)
        
        with open(self.out_test, 'w') as f_test:
            for triple in test_likes:
                f_test.write("%s\t%s\t%s\n" % triple)
        
        return train_likes
    
    def save_kg(self, train_likes):
        with open(self.out_kg, 'w') as f:
            for triple in train_likes + self.kg_triples:
                f.write("%s\t%s\t%s\n" % triple)
    
    def build_kg(self):
        self.load_users()
        self.load_movies()
        self.load_ratings()
        train_likes = self.split_sets()
        self.save_kg(train_likes)
        print("Finished building KG.")

    def load_kg_file(self):
        triples = []
        with open(self.out_kg, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    triples.append(tuple(parts))
        return triples