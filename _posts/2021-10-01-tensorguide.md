---
title: '[Tensor] 텐서 소개'
categories:
    - Tensor

tag:
    - Python
    - Tensor
    - DL

last_modified_at: 2021-10-01T14:00:00+09:00
use_math: true
comments: true
toc: true
---

이 포스트는 tensorflow 홈페이지 https://www.tensorflow.org/guide/tensor 내용을 번역하여 정리한 것 입니다. 

참고는 하되, 영문 내용을 통해 이해하는 것이 더욱 좋습니다.

# 텐서 소개


```python
import tensorflow as tf
import numpy as np
```

텐서는 `dtype` 이라고 하는 균일한 유형을 가진 다차원 배열입니다. `tf.dtypes.DType`에서 지원되는 모든 `dtype`을 볼 수 있습니다.

만약 [NumPy](https://numpy.org/devdocs/user/quickstart.html)에 익숙하시다면 텐서는 일종의 `np.arrays`와 비슷하다고 생각하면 됩니다.

모든 텐서는 Python의 숫자 및 문자열과 같이 불변합니다. 텐서의 내용을 업데이트할 수 없으며 오직 새로운 텐서만 생성할 수 있습니다.

## 기초

먼저 기본적인 텐서를 한번 만들어 봅시다.

밑에 보이는 것은 **"scalar"** 또는 **"rank-0"** 텐서 입니다. 스칼라(scalar)는 단일 값을 포함하며 "축(axes)"은 포함하지 않습니다.

**※ 기본적으로 텐서플로우에 값을 할당하기 위해서는 `constant`라는 키워드를 사용합니다.**


```python
# 이 텐서는 기본적으로 int32 텐서가 됩니다. 아래 "dtype"을 참조하세요.
rank_0_tensor = tf.constant(4) # 단일 값 4를 갖는 텐서
```


```python
rank_0_tensor
```




    <tf.Tensor: shape=(), dtype=int32, numpy=4>



**"vector"** 또는 **"rank-1"** 텐서는 값(values)의 리스트(list)와 같습니다. 벡터(vector)는 하나의 축을 가집니다.


```python
# 이번에는 "dtype"이 float인 텐서를 만들어 봅시다.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0]) # 소수점 뒤 0은 생략이 가능합니다.
```


```python
rank_1_tensor
```




    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([2., 3., 4.], dtype=float32)>



**"matrix"** 또는 **"rank-2"** 텐서는 두개의 축을 가집니다.


```python
# type을 명확하게 하고싶다면, 작성 시 "dtype"을 설정할 수도 있습니다. (아래 코드 참조)
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16) # "dtype"을 "float16"으로 설정
```


```python
rank_2_tensor
```




    <tf.Tensor: shape=(3, 2), dtype=float16, numpy=
    array([[1., 2.],
           [3., 4.],
           [5., 6.]], dtype=float16)>



밑에 개념을 알기 쉽게 사진으로 설명되어 있습니다.

<table>
<tr>
  <th>scalar, shape: <code>[]</code></th>
  <th>vector, shape: <code>[3]</code></th>
  <th>matrix, shape: <code>[3, 2]</code></th>
</tr>
<tr>
  <td>
   <img src="https://github.com/Vest1ge/Tensor/blob/main/img/scalar.png?raw=1">
  </td>

  <td>
   <img src="https://github.com/Vest1ge/Tensor/blob/main/img/vector.png?raw=1">
  </td>
  <td>
   <img src="https://github.com/Vest1ge/Tensor/blob/main/img/matrix.png?raw=1" alt="각 셀에 숫자가 포함된 3x2 그리드.">
  </td>
</tr>
</table>


텐서는 더 많은 축을 가질 수 있습니다. 다음은 세 개의 축을 가진 텐서입니다.


```python
# 밑의 코드처럼 임의적으로 지정한 축이 있을 수 있습니다.(축은 차원이라고도 합니다).
rank_3_tensor = tf.constant([
                             [[0, 1, 2, 3, 4],
                              [5, 6, 7, 8, 9]],
                             [[10, 11, 12, 13, 14],
                              [15, 16, 17, 18, 19]],
                             [[20, 21, 22, 23, 24],
                              [25, 26, 27, 28, 29]]])
```


```python
rank_3_tensor
```




    <tf.Tensor: shape=(3, 2, 5), dtype=int32, numpy=
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
    
           [[20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29]]], dtype=int32)>



두 개 이상의 축을 가지고 있는 텐서를 시각화하는 방법에는 여러 가지 방법이 있습니다.

<table>
<tr>
  <th colspan=3>3축 텐서, shape: <code>[3, 2, 5]</code></th>
<tr>
<tr>
  <td>
   <img src="https://github.com/Vest1ge/Tensor/blob/main/img/3-axis_numpy.png?raw=1"/>
  </td>
  <td>
   <img src="https://github.com/Vest1ge/Tensor/blob/main/img/3-axis_front.png?raw=1"/>
  </td>

  <td>
   <img src="https://github.com/Vest1ge/Tensor/blob/main/img/3-axis_block.png?raw=1"/>
  </td>
</tr>

</table>

`np.array` 또는 `tensor.numpy` 메소드를 사용하면 Tensorflow의 배열을 NumPy 배열로 변환할 수 있습니다.


```python
np.array(rank_2_tensor)
```




    array([[1., 2.],
           [3., 4.],
           [5., 6.]], dtype=float16)




```python
rank_2_tensor.numpy()
```




    array([[1., 2.],
           [3., 4.],
           [5., 6.]], dtype=float16)



텐서는 거의 대부분이 float와 int 타입이지만, 다음을 포함한 다른 타입도 존재합니다.

* 복소수 (complex numbers)
* 문자열 (strings)

기본 `tf.Tensor` 클래스에서는 텐서가 "직사각형",

즉, 각 축을 따라 모든 원소의 크기가 동일해야 합니다.

그러나 다양한 형태를 처리할 수 있는 특수한 타입의 텐서도 존재합니다.

* 비정형 텐서(Ragged tensors) ([링크참조](https://www.tensorflow.org/guide/ragged_tensor))
* 희소 텐서(Sparse tensors) ([링크참조](https://www.tensorflow.org/guide/sparse_tensor))

Tensorflow를 통해 덧셈, 원소별 곱셈 및 행렬 곱셈을 포함하여 텐서에 대한 기본적인 계산을 수행할 수 있습니다.



```python
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) 
# `tf.ones([2,2])` 코드를 이용해도 b와 똑같은 행렬이 나온다. (원소 값이 모두 1인 2x2 행렬)

print(tf.add(a, b), "\n") # 원소별 덧셈
print(tf.multiply(a, b), "\n") # 원소별 곱셈
print(tf.matmul(a, b), "\n") # 행렬 곱셈
```

    tf.Tensor(
    [[2 3]
     [4 5]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[3 3]
     [7 7]], shape=(2, 2), dtype=int32) 
    


이 방법을 사용할 수도 있습니다.


```python
print(a + b, "\n") # 원소별 덧셈
print(a * b, "\n")  # 원소별 곱셈
print(a @ b, "\n") # 행렬 곱셈
```

    tf.Tensor(
    [[2 3]
     [4 5]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[3 3]
     [7 7]], shape=(2, 2), dtype=int32) 
    


텐서는 모든 종류의 연산(ops)작업에 사용됩니다.


```python
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# 가장 큰 값 찾기
print(tf.reduce_max(c))
# 가장 큰 값의 인덱스 찾기
print(tf.argmax(c))
# 소프트맥스 계산
print(tf.nn.softmax(c))
```

    tf.Tensor(10.0, shape=(), dtype=float32)
    tf.Tensor([1 0], shape=(2,), dtype=int64)
    tf.Tensor(
    [[2.6894143e-01 7.3105854e-01]
     [9.9987662e-01 1.2339458e-04]], shape=(2, 2), dtype=float32)


## 형상(shapes) 에 대하여

텐서는 **shapes**를 가지고 있습니다. 사용되는 일부 용어는 다음과 같습니다.

* **형상(Shape)**: 텐서의 각 축의 길이(원소의 수)입니다.
* **순위(Rank)**: 텐서 축의 수입니다.  
 예시) rank가 0인 "scalar", rank가 1인 "vector", rank가 2인 "matrix".
* **축(Axis)** 또는 **차원(Dimension)**: 텐서의 특정한 차원
* **크기(Size)**: 텐서의 총 항목 수. 곱 형상의 벡터


참고: "2차원 텐서"에 대한 참조가 있을 수 있지만, rank-2 텐서는 일반적으로 2차원 공간을 설명하지 않습니다.

텐서와 `tf.TensorShape` 개체는 다음 항목에 액세스할 수 있는 편리한 속성을 가지고 있습니다.


```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])
```

<table>
<tr>
  <th colspan=2>rank-4 텐서, shape: <code>[3, 2, 4, 5]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/shape.png?raw=1" alt="A tensor shape is like a vector.">
    <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/4-axis_block.png?raw=1" alt="A 4-axis tensor">
  </td>
  </tr>
</table>



```python
print("모든 원소의 타입:", rank_4_tensor.dtype)
print("축(차원)의 수:", rank_4_tensor.ndim)
print("텐서의 형상(shape):", rank_4_tensor.shape)
print("축(차원)0을 따르는 원소:", rank_4_tensor.shape[0])
print("마지막 축(차원)을 따르는 원소:", rank_4_tensor.shape[-1])
print("총 원소의 수(3*2*4*5): ", tf.size(rank_4_tensor).numpy())
```

    모든 원소의 타입: <dtype: 'float32'>
    축(차원)의 수: 4
    텐서의 형상(shape): (3, 2, 4, 5)
    축(차원)0을 따르는 원소: 3
    마지막 축(차원)을 따르는 원소: 5
    총 원소의 수(3*2*4*5):  120


While axes are often referred to by their indices, you should always keep track of the meaning of each. Often axes are ordered from global to local: The batch axis first, followed by spatial dimensions, and features for each location last. This way feature vectors are contiguous regions of memory.

축은 종종 인덱스로 참조되지만 각 축의 의미를 항상 추적해야 합니다. 축은 대개 전역에서 로컬 순서로 정렬됩니다. 배치 축에 이어 공간 차원 및 각 위치의 특성이 맨 마지막에 옵니다. 이러한 방식으로 특성 벡터(feature vectors)는 메모리의 연속적인 영역입니다.

<table>
<tr>
<th>일반적인 축 순서</th>
</tr>
<tr>
    <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/shape2.png?raw=1" alt="각 축이 무엇인지 추적합니다. 4축 텐서는 배치, 너비, 높이, 특징일 수 있습니다.">
  </td>
</tr>
</table>

## 인덱싱

### 단일 축 인덱싱(Single-axis indexing)

Tensorflow는 [파이썬의 목록이나 문자열을 인덱싱하는 것](https://docs.python.org/3/tutorial/introduction.html#strings)과 유사한 표준 파이썬 인덱싱 규칙과 NumPy 인덱싱의 기본 규칙을 따릅니다.
* 인덱스는 `0`에서 부터 시작합니다.
* 음수 인덱스는 끝에서부터 거꾸로 계산합니다.
* 콜론 `:`은 슬라이스에 사용됩니다. `start:stop:step`



```python
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
rank_1_tensor.numpy()
```




    array([ 0,  1,  1,  2,  3,  5,  8, 13, 21, 34], dtype=int32)



스칼라를 사용하여 인덱싱하면 축이 제거됩니다.


```python
print("0번 인덱스(1번째 원소):", rank_1_tensor[0].numpy())
print("1번 인덱스(2번째 원소):", rank_1_tensor[1].numpy())
print("마지막 원소:", rank_1_tensor[-1].numpy())
```

    0번 인덱스(1번째 원소): 0
    1번 인덱스(2번째 원소): 1
    마지막 원소: 34


`:` 슬라이스를 사용하여 인덱싱하면 축이 유지 됩니다.


```python
print("모든 원소:", rank_1_tensor[:].numpy())
print("4번 인덱스 전 모든 원소:", rank_1_tensor[:4].numpy())
print("4번 인덱스부터 끝까지:", rank_1_tensor[4:].numpy())
print("2번 인덱스부터 7번 인덱스까지:", rank_1_tensor[2:7].numpy())
print("2 인덱스씩 건너뛰며:", rank_1_tensor[::2].numpy())
print("원소 거꾸로 정렬:", rank_1_tensor[::-1].numpy())
```

    모든 원소: [ 0  1  1  2  3  5  8 13 21 34]
    4번 인덱스 전 모든 원소: [0 1 1 2]
    4번 인덱스부터 끝까지: [ 3  5  8 13 21 34]
    2번 인덱스부터 7번 인덱스까지: [1 2 3 5 8]
    2 인덱스씩 건너뛰며: [ 0  1  3  8 21]
    원소 거꾸로 정렬: [34 21 13  8  5  3  2  1  1  0]


### 다차원 인덱싱(Multi-axis indexing)

상위 텐서는 여러 인덱스를 전달하여 인덱싱됩니다.

단일 축 사례와 정확히 동일한 규칙이 각 축에 독립적으로 적용됩니다.


```python
print(rank_2_tensor.numpy())
```

    [[1. 2.]
     [3. 4.]
     [5. 6.]]


각 인덱스에 대해 정수를 전달하면 결과는 스칼라(scalar)로 나옵니다.


```python
# rank-2 텐서에서 단일 값을 추출
print(rank_2_tensor[1, 1].numpy())
```

    4.0


다음과 같은 정수 및 `:`슬라이스의 조합을 사용하여 인덱싱할 수 있습니다.


```python
# 행 과 열 텐서 가져오기
print("두번째 행:", rank_2_tensor[1, :].numpy())
print("두번째 열:", rank_2_tensor[:, 1].numpy())
print("마지막 행:", rank_2_tensor[-1, :].numpy())
print("마지막 열의 첫번째 원소:", rank_2_tensor[0, -1].numpy())
print("첫번째 열 제외:")
print(rank_2_tensor[1:, :].numpy(), "\n")
```

    두번째 행: [3. 4.]
    두번째 열: [2. 4. 6.]
    마지막 행: [5. 6.]
    마지막 열의 첫번째 원소: 2.0
    첫번째 열 제외:
    [[3. 4.]
     [5. 6.]] 
    


3차원 텐서의 예는 다음과 같다.


```python
print(rank_3_tensor[:, :, 4])
```

    tf.Tensor(
    [[ 4  9]
     [14 19]
     [24 29]], shape=(3, 2), dtype=int32)


<table>
<tr>
<th colspan=2>배치에서 각 예의 모든 위치에서 마지막 특성 선택하기 </th>
</tr>
<tr>
    <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/index1.png?raw=1" alt="마지막 축의 인덱스-4에서 모든 값이 선택된 3x2x5 텐서.">
  </td>
      <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/index2.png?raw=1" alt="선택한 값은 2축 텐서에 패키지된다.">
  </td>
</tr>
</table>

[텐서 슬라이싱 가이드](https://tensorflow.org/guide/tensor_slicing)를 읽고 인덱싱을 적용하여 텐서의 개별 원소를 조작하는 방법에 대해 알아본다.

## 형상(Shapes) 조작하기

텐서의 형상을 바꾸는 것은 매우 유용합니다.



```python
# 형상(Shape)는 각 축의 크기를 표시하는 'TensorShape' 개체를 반환합니다.
x = tf.constant([[1], [2], [3]])
print(x.shape)
```

    (3, 1)



```python
# 이 개체를 파이썬 리스트로 변환할 수도 있습니다.
print(x.shape.as_list())
```

    [3, 1]


텐서를 새 형상으로 재구성할 수 있습니다. `tf.reshape` 는 기본 데이터를 복제할 필요가 없어 재구성이 빠릅니다.


```python
# 텐서를 새 형상으로 재구성할 수 있습니다.
# 리스트를 전달한다는 점에 유의하세요.
reshaped = tf.reshape(x, [1, 3])
```


```python
print(x.shape)
print(reshaped.shape)
```

    (3, 1)
    (1, 3)


데이터는 메모리에 레이아웃을 유지하고 요청한 형상이 동일한 데이터를 가리키는 새로운 텐서가 생성됩니다. TensorFlow는 C 스타일의 "행 중심" 메모리 순서를 사용하며, 여기서 가장 오른쪽 인덱스를 증가시키는 것은 메모리의 단일 단계에 해당합니다.


```python
print(rank_3_tensor)
```

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
    
     [[10 11 12 13 14]
      [15 16 17 18 19]]
    
     [[20 21 22 23 24]
      [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)


텐서의 평탄화 하면 메모리에 배열된 순서를 알 수 있습니다.


```python
# 특수 값 -1이면 전체 크기가 일정하게 유지되도록 해당 차원의 크기가 계산됩니다.
print(tf.reshape(rank_3_tensor, [-1]))
```

    tf.Tensor(
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29], shape=(30,), dtype=int32)


Typically the only reasonable use of `tf.reshape` is to combine or split adjacent axes (or add/remove `1`s).

일반적으로 `tf.reshape`의 합리적인 용도는 인접한 축을 결합하거나 분할하는 것(또는 `1`을 추가/제거하는 것)뿐이다.

For this 3x2x5 tensor, reshaping to (3x2)x5 or 3x(2x5) are both reasonable things to do, as the slices do not mix:

이 3x2x5 텐서의 경우 (3x2)x5 또는 3x(2x5)로 재구성하는 것이 슬라이스가 섞이지 않기 때문에 합리적인 재구성 방법이라고 할 수 있습니다.


```python
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))
```

    tf.Tensor(
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]
     [25 26 27 28 29]], shape=(6, 5), dtype=int32) 
    
    tf.Tensor(
    [[ 0  1  2  3  4  5  6  7  8  9]
     [10 11 12 13 14 15 16 17 18 19]
     [20 21 22 23 24 25 26 27 28 29]], shape=(3, 10), dtype=int32)


<table>
<th colspan=3>
몇 가지 좋은 재구성
</th>
<tr>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/reshape-before.png?raw=1" alt="3x2x5 텐서">
  </td>
  <td>
  <img src="https://github.com/Vest1ge/Tensor/blob/main/img/reshape-good1.png?raw=1" alt="(3x2)x5로 재구성된 동일한 데이터">
  </td>
  <td>
  <img src="https://github.com/Vest1ge/Tensor/blob/main/img/reshape-good2.png?raw=1" alt="3x(2x5)로 재구성된 동일한 데이터">
  </td>
</tr>
</table>


Reshaping will "work" for any new shape with the same total number of elements, but it will not do anything useful if you do not respect the order of the axes.

전체 원소 수가 동일한 새 형상에 대해 재구성되지만 축의 순서를 고려하지 않으면 유용하게 사용할 수 없습니다.

`tf.reshape`에서 축 교환이 작동하지 않으면, `tf.transpose`를 수행하여야 합니다.



```python
# 이것은 나쁜 예시입니다.

# 형상을 재구성하면 축의 순서를 변경할 수 없습니다.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 

# 매우 지저분한 형상으로 재구성됩니다.
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# 전혀 효과가 없는 코드입니다.
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]
      [10 11 12 13 14]]
    
     [[15 16 17 18 19]
      [20 21 22 23 24]
      [25 26 27 28 29]]], shape=(2, 3, 5), dtype=int32) 
    
    tf.Tensor(
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]
     [24 25 26 27 28 29]], shape=(5, 6), dtype=int32) 
    
    InvalidArgumentError: Input to reshape is a tensor with 30 values, but the requested shape requires a multiple of 7 [Op:Reshape]


<table>
<th colspan=3>
몇가지 좋지 않은 재구성
</th>
<tr>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/reshape-bad.png?raw=1" alt="축의 순서를 변경할 수 없습니다. tf.transpose를 사용하십시오.">
  </td>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/reshape-bad4.png?raw=1" alt="Anything that mixes the slices of data together is probably wrong.">
  </td>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/reshape-bad2.png?raw=1" alt="새 형상이 정확하게 맞아야 합니다.">
  </td>
</tr>
</table>

완전히 지정되지 않은 형상에 대해서 실행할 수 있습니다. 형상에 `None`(축 길이를 알 수 없음)이 포함되거나 전체 형상에 `None`(텐서의 rank를 알 수 없음)이 포함되는 것을 말합니다.

이러한 것은 [tf.RaggedTensor](https://www.tensorflow.org/guide/ragged_tensor)를 제외하고, TensorFlow의 상징적인 그래프 빌딩 API 컨텍스트에서만 발생합니다.

* [tf.function](https://www.tensorflow.org/guide/function) 
* [keras functional API](https://www.tensorflow.org/guide/keras/functional).


## `DTypes`에 관한 추가 정보

`tf.Tensor`의 데이터 타입을 검사하기 위해, `Tensor.dtype` 속성을 사용합니다.

Python 객체에서 `tf.Tensor`를 만들 때 선택적으로 데이터 타입을 지정할 수 있습니다.

지정을 하지 않으면, TensorFlow는 데이터를 나타낼 수 있는 데이터 타입을 선택합니다. TensorFlow는 Python 정수를 `tf.int32`로, Python 부동 소수점 숫자를 `tf.float32`로 변환합니다. TensorFlow는 NumPy가 배열로 변환할 때 사용하는 것과 같은 규칙을 사용합니다.

유형별로 지정할 수 있습니다.


```python
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# uint8로 지정하면 소수점의 부분을 잃습니다.
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)
```

    tf.Tensor([2 3 4], shape=(3,), dtype=uint8)


## 브로드캐스팅(Broadcasting)

브로드캐스팅은 [NumPy의 해당 기능](https://numpy.org/doc/stable/user/basics.html)에서 차용된 개념입니다. 즉, 특정 조건에서 작은 텐서가 결합 연산을 실행할 때 자동으로 "확장(streched)"되어 더 큰 텐서에 맞게되는 것을 말합니다.

가장 간단하고 일반적인 경우는 스칼라(scalar)에 텐서를 곱하거나 추가하려고 할 때입니다. 

이 경우, 스칼라는 다른 인수와 동일한 형상으로 브로드캐스트됩니다.


```python
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# 밑에 있는 모든 연산의 결과가 같다.
print(tf.multiply(x, 2))
print(x * y)
print(x * z)
```

    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    tf.Tensor([2 4 6], shape=(3,), dtype=int32)


Likewise, axes with length 1 can be stretched out to match the other arguments.  Both arguments can be stretched in the same computation.

마찬가지로 크기가 1인 축도 다른 인수와 일치하도록 확장할 수 있습니다. 두 인수 모두 동일한 계산으로 확장할 수 있습니다.

In this case a 3x1 matrix is element-wise multiplied by a 1x4 matrix to produce a 3x4 matrix. Note how the leading 1 is optional: The shape of y is `[4]`.

이 경우, 3x1 행렬에 1x4 행렬을 원소별 곱셈하면 3x4 행렬이 생성됩니다. 선행 1이 선택 사항인 점에 유의하세요. y의 형상은 `[4]`입니다.




```python
# 이것들은 같은 연산이다.
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))
```

    tf.Tensor(
    [[1]
     [2]
     [3]], shape=(3, 1), dtype=int32) 
    
    tf.Tensor([1 2 3 4], shape=(4,), dtype=int32) 
    
    tf.Tensor(
    [[ 1  2  3  4]
     [ 2  4  6  8]
     [ 3  6  9 12]], shape=(3, 4), dtype=int32)


<table>
<tr>
  <th>추가 시 브로드캐스팅: <code>[3, 1]</code> 와 <code>[1, 4]</code> 의 곱하기는 <code>[3, 4]</code> 입니다. </th>
</tr>
<tr>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/broadcasting.png?raw=1" alt="4x1 행렬에 3x1 행렬을 추가하면 3x4 행렬이 생성됩니다.">
  </td>
</tr>
</table>


같은 연산이지만 브로드캐스팅이 없는 연산이 여기 있습니다.


```python
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # 연산자를 다시 오버로딩
```

    tf.Tensor(
    [[ 1  2  3  4]
     [ 2  4  6  8]
     [ 3  6  9 12]], shape=(3, 4), dtype=int32)


브로드캐스팅은 브로드캐스트 연산으로 메모리의 확장된 텐서를 구체화하지 않기 때문에 대부분의 경우 시간과 공간적으로 모두 효율적입니다.

`tf.broadcast_to`를 사용하면 어떤 모습을 하고있는지 알 수 있습니다.


```python
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
```

    tf.Tensor(
    [[1 2 3]
     [1 2 3]
     [1 2 3]], shape=(3, 3), dtype=int32)


Unlike a mathematical op, for example, `broadcast_to` does nothing special to save memory.  Here, you are materializing the tensor.

예를 들어, 수학적 연산과 달리 `broadcast_to`는 메모리를 절약하기 위해 특별한 연산을 수행하지 않습니다. 여기에서 텐서를 구체화해봅시다.

훨씬 더 복잡해질 수 있습니다.  [해당 섹션](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) 에서는 더 많은 브로드캐스팅 트릭을 보여줍니다. (NumPy 에서)

## tf.convert_to_tensor

`tf.matmul` 및 `tf.reshape`와 같은 대부분의 ops는 클래스 `tf.Tensor`의 인수를 사용합니다. 그러나 위의 경우, 텐서 형상의 Python 객체가 수용됨을 알 수 있습니다.

전부는 아니지만 대부분의 ops는 텐서가 아닌 인수에 대해 `convert_to_tensor`를 호출합니다. 변환 레지스트리가 있어 NumPy의 `ndarray`, `TensorShape` , Python 목록 및 `tf.Variable`과 같은 대부분의 객체 클래스는 모두 자동으로 변환됩니다.

자세한 내용은 [`tf.register_tensor_conversion_function`](https://www.tensorflow.org/api_docs/python/tf/register_tensor_conversion_function)을 참조하세요. 자신만의 유형이 있으면 자동으로 텐서로 변환할 수 있습니다.


## 비정형 텐서(Ragged Tensors)

어떤 축을 따라 다양한 수의 요소를 가진 텐서를 "비정형(ragged)"이라고 합니다. 비정형 데이터에는 `tf.ragged.RaggedTensor`를 사용합니다.

예를 들어, 비정형 텐서는 정규 텐서로 표현할 수 없습니다.

<table>
<tr>
  <th>A `tf.RaggedTensor`, shape: <code>[4, None]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/ragged.png?raw=1" alt="2축 비정형 텐서는 각 행의 길이가 다를 수 있습니다.">
  </td>
</tr>
</table>


```python
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
```


```python
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

    ValueError: Can't convert non-rectangular Python sequence to Tensor.


대신 `tf.ragged.constant`를 사용하여 `tf.RaggedTensor`를 작성합니다.


```python
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
```

    <tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>


`tf.RaggedTensor`의 형상에는 알 수 없는 길이의 일부 축이 포함됩니다.


```python
print(ragged_tensor.shape)
```

    (4, None)


## 문자열 텐서(String tensors)

`tf.string`은 `dtype`이며, 텐서에서 문자열(가변 길이의 바이트 배열)과 같은 데이터를 나타낼 수 있습니다.

문자열은 원자성이므로 Python 문자열과 같은 방식으로 인덱싱할 수 없습니다. 문자열의 길이는 텐서의 축 중의 하나가 아닙니다. 문자열을 조작하는 함수에 대해서는 [`tf.strings`](https://www.tensorflow.org/api_docs/python/tf/strings)를 참조하세요.

다음은 스칼라 문자열 텐서입니다.


```python
# 텐서는 문자열이 될 수 있으며 여기에 스칼라 문자열이 있습니다.
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)
```

    tf.Tensor(b'Gray wolf', shape=(), dtype=string)


문자열의 벡터는 다음과 같습니다.

<table>
<tr>
  <th>문자열 벡터, shape: <code>[3,]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/strings.png?raw=1" alt="문자열의 길이는 텐서의 축 중 하나가 아니다.">
  </td>
</tr>
</table>


```python
# 길이가 다른 문자열 텐서가 세 개가 있다.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
# 형상이 (3, )입니다. 문자열 길이가 포함되지 않았습니다.
print(tensor_of_strings)
```

    tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)


위의 출력에서 `b` 접두사는 `tf.string dtype`이 유니코드 문자열이 아니라 바이트 문자열임을 나타냅니다. TensorFlow에서 유니코드 텍스트를 처리하는 자세한 내용은 [유니코드 튜토리얼](https://www.tensorflow.org/tutorials/load_data/unicode)을 참조하세요.

유니코드 문자를 전달하면 UTF-8로 인코딩됩니다.


```python
tf.constant("🥳👍")
```




    <tf.Tensor: shape=(), dtype=string, numpy=b'\xf0\x9f\xa5\xb3\xf0\x9f\x91\x8d'>



문자열이 있는 일부 기본 함수는 `tf.strings`을 포함하여 `tf.strings.split`에서 찾을 수 있습니다.


```python
# 분할을 사용하여 문자열을 텐서 세트로 분할할 수 있습니다.
print(tf.strings.split(scalar_string_tensor, sep=" "))
```

    tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)



```python
# 하지만 문자열로 된 텐서를 쪼개면 `비정형텐서(RaggedTensor)`로 변합니다.
# 따라서 각 문자열은 서로 다른 수의 부분으로 분할될 수 있습니다.
print(tf.strings.split(tensor_of_strings))
```

    <tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>


<table>
<tr>
  <th>세 개의 분할된 문자열, shape: <code>[3, None]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/string-split.png?raw=1" alt="Splitting multiple strings returns a tf.RaggedTensor">
  </td>
</tr>
</table>

`tf.string.to_number`:


```python
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
```

    tf.Tensor([  1.  10. 100.], shape=(3,), dtype=float32)


`tf.cast`를 사용하여 문자열 텐서를 숫자로 변환할 수는 없지만, 바이트로 변환한 다음 숫자로 변환할 수 있습니다.


```python
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)
```

    Byte strings: tf.Tensor([b'D' b'u' b'c' b'k'], shape=(4,), dtype=string)
    Bytes: tf.Tensor([ 68 117  99 107], shape=(4,), dtype=uint8)



```python
# 또는 유니코드로 분할한 다음 디코딩합니다.
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)
```

    
    Unicode bytes: tf.Tensor(b'\xe3\x82\xa2\xe3\x83\x92\xe3\x83\xab \xf0\x9f\xa6\x86', shape=(), dtype=string)
    
    Unicode chars: tf.Tensor([b'\xe3\x82\xa2' b'\xe3\x83\x92' b'\xe3\x83\xab' b' ' b'\xf0\x9f\xa6\x86'], shape=(5,), dtype=string)
    
    Unicode values: tf.Tensor([ 12450  12498  12523     32 129414], shape=(5,), dtype=int32)


`tf.string` dtype은 TensorFlow의 모든 원시 바이트 데이터에 사용됩니다. `tf.io` 모듈에는 이미지 디코딩 및 csv 구문 분석을 포함하여 데이터를 바이트로 변환하거나 바이트에서 변환하는 함수가 포함되어 있습니다.

## 희소 텐서(Sparse tensors)

때로는 매우 넓은 임베드 공간과 같이 데이터가 희소합니다. TensorFlow는 `tf.sparse.SparseTensor` 및 관련 연산을 지원하여 희소 데이터를 효율적으로 저장합니다.

<table>
<tr>
  <th>A `tf.SparseTensor`, shape: <code>[3, 4]</code></th>
</tr>
<tr>
  <td>
<img src="https://github.com/Vest1ge/Tensor/blob/main/img/sparse.png?raw=1" alt="셀 중 두 개에만 값이 있는 3x4 그리드.">
  </td>
</tr>
</table>


```python
# 희소 텐서는 메모리 효율적 방식으로 인덱스별로 값을 저장한다.
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# 희소 텐서를 고밀도(dense) 텐서로 변환할 수 있습니다.
print(tf.sparse.to_dense(sparse_tensor))
```

    SparseTensor(indices=tf.Tensor(
    [[0 0]
     [1 2]], shape=(2, 2), dtype=int64), values=tf.Tensor([1 2], shape=(2,), dtype=int32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64)) 
    
    tf.Tensor(
    [[1 0 0 0]
     [0 0 2 0]
     [0 0 0 0]], shape=(3, 4), dtype=int32)

