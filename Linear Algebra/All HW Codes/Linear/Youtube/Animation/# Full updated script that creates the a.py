import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# User input
a = st.number_input("Matrix a11", value=1.0)
b = st.number_input("Matrix a12", value=1.0)
c = st.number_input("Matrix a21", value=0.0)
d = st.number_input("Matrix a22", value=-1.0)

matrix = np.array([[a, b], [c, d]])
st.write("Transformation Matrix:")
st.write(matrix)

# Static preview (add animation next)
fig, ax = plt.subplots()
v = np.array([2, 3])
tv = matrix @ v
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original')
ax.quiver(0, 0, tv[0], tv[1], angles='xy', scale_units='xy', scale=1, color='g', label='Transformed')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.grid(True)
ax.legend()
st.pyplot(fig)
