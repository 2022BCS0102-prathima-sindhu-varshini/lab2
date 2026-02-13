pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                echo "Installing dependencies..."
                sh 'pip3 install --break-system-packages -r requirements.txt'
            }
        }

        stage('Run Training') {
            steps {
                echo "Running model training..."
                sh 'python3 train.py'
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
