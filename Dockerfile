FROM ros:jazzy

WORKDIR /workspace
RUN mkdir -p src

COPY hand_eye_calibration src/

RUN apt-get update && \
    . /opt/ros/$ROS_DISTRO/setup.sh && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y && \
    rm -rf /var/lib/apt/lists/*



RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    cd /workspace && \
    colcon build --cmake-args \
        -DCMAKE_BUILD_TYPE=Release

RUN echo "source /workspace/install/setup.bash" >> ~/.bashrc

WORKDIR /workspace

CMD ["bash"]