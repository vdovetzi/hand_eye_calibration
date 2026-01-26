FROM ros:jazzy

WORKDIR /workspace
RUN mkdir -p src

COPY hand_eye_calibration/package.xml src/hand_eye_calibration/

RUN apt-get update && \
    . /opt/ros/$ROS_DISTRO/setup.sh && \
    rosdep update && \
    rosdep install --from-paths src -y && \
    rm -rf /var/lib/apt/lists/*

COPY hand_eye_calibration src/

RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    cd /workspace && \
    colcon build --cmake-args \
        -DCMAKE_BUILD_TYPE=Release

RUN echo "source /workspace/install/setup.bash" >> ~/.bashrc

WORKDIR /workspace

CMD ["bash"]