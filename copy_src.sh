#!/bin/bash

# Настройки подключения
ORANGE_PI_USER="pi"
ORANGE_PI_IP="192.168.137.212"
ORANGE_PI_DIR="/home/pi/gstreamer-and-nano-track/src"
LOCAL_DIR="/home/fedor/repos/nano_plus_gstreamer/src"

# sudo rm -rf ./object_classifier/target

echo "Удаляем старую папку на Orange Pi..."
ssh ${ORANGE_PI_USER}@${ORANGE_PI_IP} "rm -rf ${ORANGE_PI_DIR}"

echo "Создаём новую папку на Orange Pi..."
ssh ${ORANGE_PI_USER}@${ORANGE_PI_IP} "mkdir -p ${ORANGE_PI_DIR}"

echo "Копируем содержимое папки на Orange Pi..."
rsync -avz -e ssh ${LOCAL_DIR}/ ${ORANGE_PI_USER}@${ORANGE_PI_IP}:${ORANGE_PI_DIR}/

echo "Синхронизация завершена!"