import numpy as np
import pandas as pd
import lightgbm as lgb
import os

IDIR = os.path.abspath('../data')


print('loading prior')
newpriors = pd.read_csv(os.path.join(IDIR,'newpriors.csv'), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8,
            'user_id' : np.int32,
            'eval_set' : 'category',
            'order_number': np.int16,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'days_since_prior_order': np.float32})

print('loading users')
users = pd.read_csv(os.path.join(IDIR,'users.csv'), dtype={
            'average_days_between_orders': np.float32,
            'user_num_orders': np.int16,
            'total_num_items': np.int16,
            'distinct_num_items' : np.int16,
            'average_basket' : np.float32}, index_col = 'user_id')

print('loading train')
train = pd.read_csv(os.path.join(IDIR,'order_products__train.csv'), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(os.path.join(IDIR,'orders.csv'), dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

print('loading newproducts')
newproducts = pd.read_csv(os.path.join(IDIR,'newproducts.csv'), dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'orders' : np.int32,
        'reorders' : np.int32,
        'reorder_rate' : np.float64,
        'department_id': np.uint8})

print('loading userXproduct')           
userXproduct = pd.read_csv(os.path.join(IDIR,'userXproduct.csv'), dtype={
            'user_id': np.int32,
            'product_id': np.int32,
            'UP_num_item': np.int16,
            'UP_first_order': np.int16,
            'UP_last_order' : np.int16,
            'user_num_orders' :np.int16,
            'UP_sum_add_cart' : np.int16,
            'UP_frequency': np.float16,
            'UP_average_add_cart': np.float16})




### train / test orders ###
print('split orders : train, test')
test_orders = orders[orders.eval_set == 'test']
train_orders = orders[orders.eval_set == 'train']

train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

### build list of candidate products to reorder, with features ###





def features(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i=0
    for row in selected_orders.itertuples():
        i+=1
        if i%10000 == 0: print('order row',i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.products_list[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train.index for product in user_products]
        
    df = pd.DataFrame({'order_id':order_list, 'product_id':product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list
    
    print('user related features')
    df['user_id'] = df.order_id.map(orders.user_id)
    df['user_total_orders'] = df.user_id.map(users.user_num_orders)
    df['user_total_items'] = df.user_id.map(users.total_num_items)
    df['total_distinct_items'] = df.user_id.map(users.distinct_num_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] =  df.user_id.map(users.average_basket)
    
    print('order related features')
    # df['dow'] = df.order_id.map(orders.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
    df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders
    
    print('product related features')
    df['aisle_id'] = df.product_id.map(newproducts.aisle_id)
    df['department_id'] = df.product_id.map(newproducts.department_id)
    df['product_orders'] = df.product_id.map(newproducts.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(newproducts.reorders)
    df['product_reorder_rate'] = df.product_id.map(newproducts.reorder_rate)

    print('user_X_product related features')
    #df['z'] = df.user_id * 100000 + df.product_id
    

    df.merge(userXproduct, on = ['user_id', 'product_id'])
    df.drop(['user_id'], axis=1, inplace=True)
    #df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    #df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    #df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    #df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    #df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(lambda x: min(x, 24-x)).astype(np.int8)
    #df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
    #                                              df.order_id.map(orders.order_dow)

    #df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    print(df.dtypes)
    print(df.memory_usage())
    return (df, labels)
    

df_train, labels = features(train_orders, labels_given=True)
df_train.to_csv(os.path.join(IDIR, 'df_train.csv'), index = false)

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket',
       'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last'] # 'dow', 'UP_same_dow_as_last_order'


print('formating for lgb')
d_train = lgb.Dataset(df_train[f_to_use],
                      label=labels,
                      categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'
del df_train

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
ROUNDS = 100

print('light GBM train :-)')
bst = lgb.train(params, d_train, ROUNDS)
# lgb.plot_importance(bst, figsize=(9,20))
del d_train

### build candidates list for test ###

df_test, _ = features(test_orders)

print('light GBM predict')
preds = bst.predict(df_test[f_to_use])

df_test['pred'] = preds

TRESHOLD = 0.22  # guess, should be tuned with crossval on a subset of train data

d = dict()
for row in df_test.itertuples():
    if row.pred > TRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in test_orders.order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('sub.csv', index=False)