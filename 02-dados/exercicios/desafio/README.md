# desafio

## Requesitos
Para rodar esse projeto localmente é necessario os seguintes softwares
- Python na versão descrita em .python-version
- Docker desktop

## Setup
Todo o projeto é orquestrado pelo docker compose, logo basta um `docker-compose up` e o AirFlow estará de pé

### Issues conhecidas
- O dataset está duplicando ao rodar um novo schedule. Preciso fazer um identificador unico para o dataset de carro, provavelmente modelo-ano-região
