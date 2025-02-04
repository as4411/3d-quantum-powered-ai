import numpy as np
import plotly.graph_objects as go
import streamlit as st
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

# Quantum Circuit Setup: Define a single-qubit circuit with custom gates
def create_quantum_circuit():
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Apply Hadamard gate to put qubit in superposition
    qc.rx(np.pi / 4, 0)  # Rotate around the X-axis by pi/4
    qc.measure(0, 0)  # Measure the qubit and store the result in the classical bit
    return qc

# Function to simulate the quantum circuit and get results using Sampler
def simulate_quantum_circuit(qc):
    sampler = Sampler()
    result = sampler.run([qc]).result()
    quasi_dist = result.quasi_dists[0]
    probabilities = [quasi_dist.get(0, 0), quasi_dist.get(1, 0)]
    return probabilities

# Generate a color gradient based on quantum results
def quantum_color_generator(probabilities):
    prob_0 = probabilities[0]
    prob_1 = probabilities[1]
    color = f"rgba({int(255 * prob_0)}, {int(255 * prob_1)}, 150, 0.7)"
    return color

# Create the bicycle model
def plot_quantum_bicycle_model():
    qc = create_quantum_circuit()
    probabilities = simulate_quantum_circuit(qc)
    bike_color = quantum_color_generator(probabilities)

    # Bicycle frame
    bike_frame = go.Mesh3d(
        x=[0, 0.5, 1, 1.5, 1, 0.5, 0],
        y=[0, 0.5, 0.5, 0, -0.5, -0.5, 0],
        z=[0, 0, 0, 0, 1, 1, 1],
        color=bike_color,
        opacity=0.7,
        i=[0, 0, 0, 1, 1, 2, 3, 3],
        j=[1, 2, 4, 4, 5, 5, 6, 2],
        k=[2, 4, 3, 3, 6, 5, 1, 0],
    )

    # Wheels
    theta = np.linspace(0, 2 * np.pi, 100)
    wheel_color = quantum_color_generator(probabilities)

    # Front wheel
    front_wheel = go.Scatter3d(
        x=1.5 + 0.1 * np.cos(theta),
        y=0.5 + 0.1 * np.sin(theta),
        z=np.zeros(100),
        mode='lines',
        line=dict(color=wheel_color, width=5),
    )

    # Rear wheel
    rear_wheel = go.Scatter3d(
        x=0.5 + 0.1 * np.cos(theta),
        y=-0.5 + 0.1 * np.sin(theta),
        z=np.zeros(100),
        mode='lines',
        line=dict(color=wheel_color, width=5),
    )

    # Combine into a figure
    fig = go.Figure(data=[bike_frame, front_wheel, rear_wheel])

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 3], backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(range=[-1, 1], backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(range=[-1, 2], backgroundcolor="rgb(230, 230,230)"),
        ),
        title='Quantum-Inspired 3D Bicycle Model',
        showlegend=False
    )

    return fig

# Create the car model
def plot_quantum_car_model():
    qc = create_quantum_circuit()
    probabilities = simulate_quantum_circuit(qc)
    car_color = quantum_color_generator(probabilities)

    # Define the car body
    car_body = go.Mesh3d(
        x=[0, 2, 2, 0, 0, 2, 2, 0],
        y=[0, 0, 1, 1, 0, 0, 1, 1],
        z=[0, 0, 0, 0, 1, 1, 1, 1],
        color=car_color,
        opacity=0.7,
        i=[0, 0, 0, 4, 4, 2, 6, 7, 5, 1, 2, 6],
        j=[1, 2, 4, 5, 6, 3, 7, 6, 4, 5, 3, 7],
        k=[2, 4, 6, 7, 7, 1, 1, 0, 0, 2, 6, 0],
    )

    # Define wheels
    theta = np.linspace(0, 2 * np.pi, 100)
    wheel_color = quantum_color_generator(probabilities)

    # Front-left wheel
    wheel1 = go.Scatter3d(
        x=0.5 + 0.1 * np.cos(theta),
        y=0.1 * np.sin(theta),
        z=np.zeros(100),
        mode='lines',
        line=dict(color=wheel_color, width=5),
    )

    # Front-right wheel
    wheel2 = go.Scatter3d(
        x=1.5 + 0.1 * np.cos(theta),
        y=0.1 * np.sin(theta),
        z=np.zeros(100),
        mode='lines',
        line=dict(color=wheel_color, width=5),
    )

    # Rear-left wheel
    wheel3 = go.Scatter3d(
        x=0.5 + 0.1 * np.cos(theta),
        y=0.1 * np.sin(theta),
        z=np.ones(100),
        mode='lines',
        line=dict(color=wheel_color, width=5),
    )

    # Rear-right wheel
    wheel4 = go.Scatter3d(
        x=1.5 + 0.1 * np.cos(theta),
        y=0.1 * np.sin(theta),
        z=np.ones(100),
        mode='lines',
        line=dict(color=wheel_color, width=5),
    )

    # Combine into a figure
    fig = go.Figure(data=[car_body, wheel1, wheel2, wheel3, wheel4])

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 3], backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(range=[-1, 2], backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(range=[-1, 2], backgroundcolor="rgb(230, 230,230)"),
        ),
        title='Quantum-Inspired 3D Car Model',
        showlegend=False
    )

    return fig


# Streamlit Setup
st.title("Quantum-Inspired 3D Vehicle Models")

vehicle_choice = st.selectbox("Choose a vehicle to visualize:", ["Car", "Bicycle"])

if vehicle_choice == "Car":
    car_fig = plot_quantum_car_model()
    st.plotly_chart(car_fig)

elif vehicle_choice == "Bicycle":
    bike_fig = plot_quantum_bicycle_model()
    st.plotly_chart(bike_fig)

# Visualizing the Bloch Sphere for the circuit
st.title("Quantum State Visualization - Bloch Sphere")
qc = create_quantum_circuit()
state = Statevector.from_instruction(qc)  # Get state vector of the circuit

# Display Bloch Sphere
fig, ax = plt.subplots(figsize=(6, 6))
plot_bloch_multivector(state)
st.pyplot(fig)