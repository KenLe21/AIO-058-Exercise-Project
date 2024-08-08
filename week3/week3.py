#4. PLAY TENNIS CLASSIFIER IMPLEMENTATION
#4.1
import numpy as np
def create_train_data():
    data = [
        ['Sunny', 'Hot', 'High', 'Weak', 'no'],
        ['Sunny', 'Hot', 'High', 'Strong', 'no'],
        ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
        ['Overcast', 'Mild', 'High', 'Weak', 'no'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'yes']
    ]
    
    # Chuyển đổi danh sách thành numpy array
    return np.array(data)

train_data = create_train_data()
print(train_data)


#4.2
def compute_prior_probability(train_data):
    # Các lớp duy nhất trong cột cuối cùng của dữ liệu (PlayTennis)
    y_unique = ['no', 'yes']
    
    # Khởi tạo mảng chứa xác suất a priori cho các lớp
    prior_probability = np.zeros(len(y_unique))
    
    # Tính số lượng mẫu
    total_samples = train_data.shape[0]
    
    # Tính số lượng mẫu cho mỗi lớp
    for i, label in enumerate(y_unique):
        # Đếm số mẫu thuộc lớp hiện tại
        count = np.sum(train_data[:, -1] == label)
        # Tính xác suất a priori cho lớp hiện tại
        prior_probability[i] = count / total_samples
    
    return prior_probability

prior_probability = compute_prior_probability(train_data)
print("P(Play Tennis = No):", prior_probability[0])
print("P(Play Tennis = Yes):", prior_probability[1])


#4.3 + cau 15
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []

    num_attributes = train_data.shape[1] - 1  # Số lượng thuộc tính (không bao gồm cột lớp)

    for i in range(num_attributes):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)
        x_conditional_probability = []
        for label in y_unique:
            # Lọc dữ liệu theo lớp
            class_data = train_data[train_data[:, -1] == label]
            class_size = class_data.shape[0]
            
            # Tính xác suất điều kiện cho từng giá trị của thuộc tính
            attribute_prob = []
            for x in x_unique:
                count = np.sum(class_data[:, i] == x)
                prob = count / class_size
                attribute_prob.append(prob)
            
            x_conditional_probability.append(attribute_prob)
        
        conditional_probability.append(x_conditional_probability)
    
    return conditional_probability, list_x_name

# Tạo dữ liệu huấn luyện
train_data = create_train_data()

# Tính xác suất điều kiện
_, list_x_name = compute_conditional_probability(train_data)

# In kết quả
print("x1 = ", list_x_name[0])
print("x2 = ", list_x_name[1])
print("x3 = ", list_x_name[2])
print("x4 = ", list_x_name[3])


#4.4
import numpy as np
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []

    num_attributes = train_data.shape[1] - 1  # Số lượng thuộc tính (không bao gồm cột lớp)

    for i in range(num_attributes):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)
        
        x_conditional_probability = []

        for label in y_unique:
            # Lọc dữ liệu theo lớp
            class_data = train_data[train_data[:, -1] == label]
            class_size = class_data.shape[0]
            
            # Tính xác suất điều kiện cho từng giá trị của thuộc tính
            attribute_prob = []
            for x in x_unique:
                count = np.sum(class_data[:, i] == x)
                prob = count / class_size
                attribute_prob.append(prob)
            
            x_conditional_probability.append(attribute_prob)
        conditional_probability.append(x_conditional_probability)
    return conditional_probability, list_x_name
conditional_probability, list_x_name = compute_conditional_probability(train_data)


print("Conditional Probabilities:")
for i, probs in enumerate(conditional_probability):
    print(f"Attribute {i}:")
    for j, label_probs in enumerate(probs):
        print(f"  P(Attribute {i} | Play Tennis = {j}): {dict(zip(list_x_name[i], label_probs))}")

def get_index_from_value(feature_name, list_features):
    # Chuyển list_features thành numpy array để sử dụng np.where
    list_features_array = np.array(list_features)
    
    # Tìm chỉ số của feature_name trong list_features_array
    indices = np.where(list_features_array == feature_name)[0]
    
    # Nếu không tìm thấy, trả về -1 hoặc xử lý theo cách khác nếu cần
    if len(indices) == 0:
        raise ValueError(f"Feature name '{feature_name}' not found in list_features.")
    
    # Trả về chỉ số đầu tiên tìm thấy
    return indices[0]

# Ví dụ sử dụng hàm
list_features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
feature_name = 'Humidity'
index = get_index_from_value(feature_name, list_features)
print(f"The index of feature '{feature_name}' is: {index}")


#cau 16
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []
    num_attributes = train_data.shape[1] - 1  # Số lượng thuộc tính (không bao gồm cột lớp)
    for i in range(num_attributes):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)
        x_conditional_probability = []
        for label in y_unique:
            class_data = train_data[train_data[:, -1] == label]
            class_size = class_data.shape[0]
            attribute_prob = []
            for x in x_unique:
                count = np.sum(class_data[:, i] == x)
                prob = count / class_size
                attribute_prob.append(prob)
            
            x_conditional_probability.append(attribute_prob)
        
        conditional_probability.append(x_conditional_probability)
    
    return conditional_probability, list_x_name
train_data = create_train_data()
_, list_x_name = compute_conditional_probability(train_data)
outlook = list_x_name[0]

i1 = get_index_from_value("Overcast", outlook)
i2 = get_index_from_value("Rain", outlook)
i3 = get_index_from_value("Sunny", outlook)
print(i1, i2, i3)


#cau 17
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []

    for i in range(train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)

        # Create a dictionary to store the counts
        counts = {}
        for x in x_unique:
            counts[x] = np.sum(train_data[:, i] == x)
        
        # Calculate conditional probabilities
        prob = np.zeros((len(y_unique), len(x_unique)))
        for j, y in enumerate(y_unique):
            subset = train_data[train_data[:, -1] == y]
            for k, x in enumerate(x_unique):
                prob[j, k] = np.sum(subset[:, i] == x) / len(subset)
        conditional_probability.append(prob)

    return conditional_probability, list_x_name

# Hàm tìm chỉ số của giá trị thuộc tính trong danh sách các giá trị của thuộc tính
def get_index_from_value(feature_name, list_features):
    feature_name = feature_name.strip()
    list_features = np.array([x.strip() for x in list_features])
    indices = np.where(list_features == feature_name)[0]
    if len(indices) == 0:
        raise ValueError(f"Feature name '{feature_name}' not found in list_features.")
    return indices[0]

train_data = create_train_data()
conditional_probability, list_x_name = compute_conditional_probability(train_data)

# P("Outlook" = "Sunny" | "Play Tennis" = "Yes")
x1 = get_index_from_value('Sunny', list_x_name[0])
print("P('Outlook' = 'Sunny' | 'Play Tennis' = 'Yes') = ", np.round(conditional_probability[0][1, x1], 2))



#cau 18
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []

    for i in range(train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)

        # Tạo dictionary để lưu số lượng
        counts = {}
        for x in x_unique:
            counts[x] = np.sum(train_data[:, i] == x)
        
        # Tính xác suất điều kiện
        prob = np.zeros((len(y_unique), len(x_unique)))
        for j, y in enumerate(y_unique):
            subset = train_data[train_data[:, -1] == y]
            for k, x in enumerate(x_unique):
                prob[j, k] = np.sum(subset[:, i] == x) / len(subset)
        conditional_probability.append(prob)

    return conditional_probability, list_x_name

# Hàm tìm chỉ số của giá trị thuộc tính trong danh sách các giá trị của thuộc tính
def get_index_from_value(feature_name, list_features):
    feature_name = feature_name.strip()
    list_features = np.array([x.strip() for x in list_features])
    indices = np.where(list_features == feature_name)[0]
    if len(indices) == 0:
        raise ValueError(f"Feature name '{feature_name}' not found in list_features.")
    return indices[0]

train_data = create_train_data()
conditional_probability, list_x_name = compute_conditional_probability(train_data)

# Tính P("Outlook" = "Sunny" | "Play Tennis" = "No")
x1 = get_index_from_value('Sunny', list_x_name[0])
print("P('Outlook' = 'Sunny' | 'Play Tennis' = 'No') = ", np.round(conditional_probability[0][0, x1], 2))



#4.5
import numpy as np
# Hàm tính xác suất prior
def compute_prior_probablity(train_data):
    y_unique = ['no', 'yes']
    prior_probability = np.zeros(len(y_unique))

    total_count = train_data.shape[0]
    for i, y in enumerate(y_unique):
        prior_probability[i] = np.sum(train_data[:, -1] == y) / total_count

    return prior_probability

# Hàm tính xác suất điều kiện
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []

    for i in range(train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)

        # Tạo dictionary để lưu số lượng
        counts = {}
        for x in x_unique:
            counts[x] = np.sum(train_data[:, i] == x)
        
        # Tính xác suất điều kiện
        prob = np.zeros((len(y_unique), len(x_unique)))
        for j, y in enumerate(y_unique):
            subset = train_data[train_data[:, -1] == y]
            for k, x in enumerate(x_unique):
                prob[j, k] = np.sum(subset[:, i] == x) / len(subset)
        conditional_probability.append(prob)

    return conditional_probability, list_x_name

# Hàm huấn luyện mô hình Naive Bayes
def train_naive_bayes(train_data):
    # Tính xác suất prior
    prior_probability = compute_prior_probablity(train_data)

    # Tính xác suất điều kiện
    conditional_probability, list_x_name = compute_conditional_probability(train_data)

    return prior_probability, conditional_probability, list_x_name

train_data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(train_data)

print("Prior Probability:", prior_probability)
print("Conditional Probability:", conditional_probability)
print("List of Feature Values:", list_x_name)



#4.6
import numpy as np

def prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability):
    # Lấy chỉ số của các giá trị thuộc tính trong danh sách các giá trị
    x1 = get_index_from_value(X[0], list_x_name[0])
    x2 = get_index_from_value(X[1], list_x_name[1])
    x3 = get_index_from_value(X[2], list_x_name[2])
    x4 = get_index_from_value(X[3], list_x_name[3])

    # Tính toán xác suất cho mỗi lớp mục tiêu
    p0 = prior_probability[0]  # P("Play Tennis" = "No")
    p1 = prior_probability[1]  # P("Play Tennis" = "Yes")

    # Tính xác suất điều kiện cho từng lớp mục tiêu
    # P("Outlook" = x1 | "Play Tennis" = "No")
    # P("Temperature" = x2 | "Play Tennis" = "No")
    # P("Humidity" = x3 | "Play Tennis" = "No")
    # P("Wind" = x4 | "Play Tennis" = "No")
    p0 *= (conditional_probability[0][0, x1] *  # P(Outlook=x1 | Play Tennis=No)
           conditional_probability[1][0, x2] *  # P(Temperature=x2 | Play Tennis=No)
           conditional_probability[2][0, x3] *  # P(Humidity=x3 | Play Tennis=No)
           conditional_probability[3][0, x4])  # P(Wind=x4 | Play Tennis=No)

    # P("Outlook" = x1 | "Play Tennis" = "Yes")
    # P("Temperature" = x2 | "Play Tennis" = "Yes")
    # P("Humidity" = x3 | "Play Tennis" = "Yes")
    # P("Wind" = x4 | "Play Tennis" = "Yes")
    p1 *= (conditional_probability[0][1, x1] *  # P(Outlook=x1 | Play Tennis=Yes)
           conditional_probability[1][1, x2] *  # P(Temperature=x2 | Play Tennis=Yes)
           conditional_probability[2][1, x3] *  # P(Humidity=x3 | Play Tennis=Yes)
           conditional_probability[3][1, x4])  # P(Wind=x4 | Play Tennis=Yes)

    # Dự đoán lớp mục tiêu
    if p0 > p1:
        y_pred = 'no'
    else:
        y_pred = 'yes'

    return y_pred

train_data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(train_data)

# Dữ liệu kiểm tra D11
X_test = ['Sunny', 'Cool', 'High', 'Strong']
prediction = prediction_play_tennis(X_test, list_x_name, prior_probability, conditional_probability)

print("predict for D11:", prediction)



#5.(OPTIONAL) IRIS CLASSIFIER IMPLEMENTATION
import numpy as np

def create_train_data_iris():
    # Đọc dữ liệu từ file dưới dạng chuỗi
    data = np.genfromtxt('week3/iris.data.txt', delimiter=',', dtype=str)
    
    # Chia dữ liệu thành các thuộc tính và nhãn
    features = data[:, :-1]  # Các cột thuộc tính từ cột 0 đến cột cuối cùng - 1
    labels = data[:, -1]     # Cột nhãn là cột cuối cùng
    
    # Chuyển đổi các thuộc tính thành kiểu số (float)
    features = features.astype(float)
    
    # Ghép thuộc tính và nhãn vào dữ liệu huấn luyện
    train_data = np.column_stack((features, labels))
    
    return train_data

# Kiểm tra kết quả
train_data = create_train_data_iris()
print("Train data shape:", train_data.shape)
print("Sample data:", train_data[:5])


from scipy.stats import norm
def train_gaussian_naive_bayes(train_data):
    # Tính toán các lớp duy nhất
    y_unique = np.unique(train_data[:, -1])
    
    prior_probability = {}
    conditional_probability = {}
    
    for label in y_unique:
        # Lọc dữ liệu theo lớp
        class_data = train_data[train_data[:, -1] == label]
        class_features = class_data[:, :-1].astype(float)  # Chuyển đổi thành float
        
        # Tính xác suất prior
        prior_probability[label] = len(class_data) / len(train_data)
        
        # Tính toán các tham số Gaussian cho từng thuộc tính
        means = np.mean(class_features, axis=0)
        stds = np.std(class_features, axis=0)
        
        conditional_probability[label] = (means, stds)
    
    return prior_probability, conditional_probability

def prediction_iris(X, prior_probability, conditional_probability):
    y_unique = list(prior_probability.keys())
    
    # Tính toán xác suất cho mỗi lớp
    posteriors = {}
    
    for label in y_unique:
        mean, std = conditional_probability[label]
        
        # Xác suất prior
        prior = prior_probability[label]
        
        # Xác suất conditional (Gaussian)
        likelihood = np.prod(norm.pdf(X, mean, std))
        
        # Xác suất posterior
        posteriors[label] = prior * likelihood
    
    # Lớp có xác suất posterior cao nhất
    return y_unique.index(max(posteriors, key=posteriors.get))

# Ví dụ test
X = [6.3, 3.3, 6.0, 2.5]
train_data = create_train_data_iris()
prior_probability, conditional_probability = train_gaussian_naive_bayes(train_data)
y_unique = np.unique(train_data[:, -1])
pred = y_unique[prediction_iris(X, prior_probability, conditional_probability)]
print(f"Prediction: {pred}")

X = [5.0 ,2.0 ,3.5 ,1.0]
train_data = create_train_data_iris()
prior_probability, conditional_probability = train_gaussian_naive_bayes(train_data)
y_unique = np.unique(train_data[:, -1])
pred = y_unique[prediction_iris(X, prior_probability, conditional_probability)]
print(f"Prediction: {pred}")


X = [4.9 ,3.1 ,1.5 ,0.1]
train_data = create_train_data_iris()
prior_probability, conditional_probability = train_gaussian_naive_bayes(train_data)
y_unique = np.unique(train_data[:, -1])
pred = y_unique[prediction_iris(X, prior_probability, conditional_probability)]
print(f"Prediction: {pred}")

