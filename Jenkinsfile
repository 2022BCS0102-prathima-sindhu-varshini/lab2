pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                echo "Cloning repository..."
            }
        }

        stage('Install Dependencies') {
            steps {
                echo "Installing dependencies..."
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Run Training') {
            steps {
                echo "Running model training..."
                bat 'python train.py'
            }
        }

        stage('Print Student Details') {
            steps {
                echo "Student Name: Sabbi Prathima Sindhu Varshini"
                echo "Roll Number: 2022BCS0102"
            }
        }
    }
}
