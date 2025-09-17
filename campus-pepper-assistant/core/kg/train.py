from core.kg.rotatE import RotatEModel
import tensorflow as tf
import numpy as np
import random


def batch_iter(data, batch_size, shuffle=True):
    data = list(data)
    if shuffle:
        random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def load_triples(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def generate_negative_samples(triples, num_entities):
    neg_triples = []
    for h, r, t in triples:
        corrupt_tail = (h, r, np.random.randint(num_entities))
        corrupt_head = (np.random.randint(num_entities), r, t)
        neg_triples.append(corrupt_tail)
        neg_triples.append(corrupt_head)
    return neg_triples

def build_vocab(files):
    entity2id, relation2id = {}, {}
    eid = 0
    rid = 0
    for file in files:
        with open(file) as f:
            for line in f:
                h, r, t = line.strip().split("\t")
                if h not in entity2id:
                    entity2id[h] = eid
                    eid += 1
                if t not in entity2id:
                    entity2id[t] = eid
                    eid += 1
                if r not in relation2id:
                    relation2id[r] = rid
                    rid += 1
    return entity2id, relation2id

def evaluate(model, sess, test_triples, entity2id, relation2id, id2entity, k=10):
    hits_at_k = 0.0
    mrr = 0.0
    total = 0
    course_id = [eid for ent, eid in entity2id.items() if ent.startswith("course_")]

    for h, r, t in test_triples:
        scores = model.get_score_op(sess, h, r, course_id)
        scores = list(scores)
        true_score = model.get_score_op(sess, h, r, [t])[0]

        # Rank is 1 + #items with higher score
        rank = 1 + sum([1 for s in scores if s > true_score])
        mrr += 1.0 / rank
        if rank <= k:
            hits_at_k += 1
        total += 1

    print("\n--- Evaluation ---")
    print("MRR: %.4f" % (mrr / total))
    print("Hits@%d: %.4f" % (k, hits_at_k / total))

def recommend_top_k(user_str, model, sess, entity2id, relation2id, id2entity, k=5):
    user_id = entity2id.get(user_str)
    relation_id = relation2id.get("likes")
    course_ids = [eid for ent, eid in entity2id.items() if ent.startswith("course_")]

    scores = model.get_score_op(sess, user_id, relation_id, course_ids)
    top_indices = np.argsort(scores)[-k:][::-1]
    top_course_ids = [course_ids[i] for i in top_indices]
    return [id2entity[mid] for mid in top_course_ids]

def train_rotate():
    entity2id, relation2id = build_vocab(["data/kg.txt"])
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    train_triples = load_triples("data/train.txt", entity2id, relation2id)
    test_triples = load_triples("data/test.txt", entity2id, relation2id)

    model = RotatEModel(num_entities=len(entity2id), num_relations=len(relation2id), embedding_dim=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 10000  

    for epoch in range(40):
        # Shuffle training triples each epoch
        for batch_triples in batch_iter(train_triples, batch_size):
            neg_batch = generate_negative_samples(batch_triples, len(entity2id))
            
            all_batch = batch_triples + neg_batch
            labels = [1.0] * len(batch_triples) + [-1.0] * len(neg_batch)

            heads = [h for h, r, t in all_batch]
            rels = [r for h, r, t in all_batch]
            tails = [t for h, r, t in all_batch]

            feed_dict = {
                model.heads: heads,
                model.relations: rels,
                model.tails: tails,
                model.labels: labels
            }

            loss, _ = sess.run([model.loss, model.optimizer], feed_dict=feed_dict)

        if epoch % 10 == 0:
            print("Epoch %d - Loss: %.4f" % (epoch, loss))

    saver = model.get_saver()
    saver.save(sess, "checkpoints/rotate_model.ckpt")
    print("Model saved to checkpoints/rotate_model.ckpt")
