## GitHub Description

A containerized ontology similarity web application powered by Docker and Kubernetes. It provides APIs and an interactive frontend for computing and comparing semantic similarities between ontology concepts using Shortest Path / Node2Vec and Hierarchical Structural Similarity algorithms.

---

# README.md

# Ontology Similarity Web Application (Docker + Kubernetes Edition)

This project provides a containerized, interactive web-based platform for querying semantic similarities between ontology concepts (e.g., diseases, taxa, or biological classifications) using advanced graph algorithms:

- **Shortest Path / Node2Vec Embedding**: Measures relational similarity across an ontology graph using graph distances or learned embeddings.
- **Hierarchical Structural Similarity (HSS)**: Analyzes local hierarchical similarity based on ancestor/descendant overlap and depth.

## 🚀 Features

- RESTful API with FastAPI for similarity querying
- Simple web frontend for interactive user queries
- Containerized with Docker for consistent deployment
- Kubernetes-ready manifests for scalable deployment

## 🛠 Technology Stack

- **Backend:** Python (FastAPI, NetworkX, Node2Vec)
- **Frontend:** HTML + JavaScript (minimal or optional React/Vue)
- **Containerization:** Docker
- **Orchestration:** Kubernetes (k8s)

## 🗂 Project Structure

```
ontology-similarity/
├── main.py               # FastAPI backend API
├── index.html            # Frontend HTML interface
├── Dockerfile            # Docker container definition
├── ontology-app.yaml     # Kubernetes deployment + service
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🔧 Setup and Run Locally with Docker

### 1. Build Docker Image

```bash
docker build -t ontology-similarity-api .
```

### 2. Run the Container

```bash
docker run -d -p 8000:8000 ontology-similarity-api
```

### 3. Open Frontend

Simply open `index.html` in your browser and interact with the API at `http://localhost:8000`.

---

## ☸️ Deploy with Kubernetes

### 1. Apply Deployment and Service

```bash
kubectl apply -f ontology-app.yaml
```

### 2. Check Status

```bash
kubectl get pods
kubectl get services
```

If you're using Minikube:

```bash
minikube service ontology-similarity-service
```

---

## 🚢 Deployment Options

You can deploy this app to any Kubernetes-compatible platform:

- [AWS EKS](https://aws.amazon.com/eks/)
- [Google GKE](https://cloud.google.com/kubernetes-engine)
- [Azure AKS](https://azure.microsoft.com/en-au/products/kubernetes-service/)
- [DigitalOcean Kubernetes](https://www.digitalocean.com/products/kubernetes)
- [Minikube](https://minikube.sigs.k8s.io/docs/)

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Enjoy exploring semantic similarities with your ontology data — now scalable with Docker and Kubernetes! 🚀☸️

i
