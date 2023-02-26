# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under The 3-Clause BSD License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from multi_robomaster import multi_robot


def base_action_1(robot_group,abc):
    robot_group.mission_pad_on()
    #robot_group.mon()
    robot_group.takeoff().wait_for_completed()
    robot_group.go({1: [50, 50, 100, 50, "m1"], 2: [-50, 50, 100, 50,"m12"]}).wait_for_completed()
    robot_group.set_mled_char("r", "heart")
    robot_group.go({1: [50, -50, 100, 50,"m1"], 2: [50, 50, 100, 50,"m12"]}).wait_for_completed()
    robot_group.set_mled_char("p", "heart")
    robot_group.go({1: [-50, -50, 100, 50, "m1"], 2: [50, -50, 100, 50,"m12"]}).wait_for_completed()
    robot_group.go({1: [-50, 50, 100, 50, "m1"], 2: [-50, -50, 100, 50, "m12"]}).wait_for_completed()
    robot_group.land().wait_for_completed()
    robot_group.mission_pad_off()
    #robot_group.moff()


if __name__ == '__main__':
    multi_drone = multi_robot.MultiDrone()
    multi_drone.initialize(robot_num=2)
    # get drone sn by run the expamles of /15_multi_robot/multi_drone/01_scan_ip.py
    #drone_ip_list = multi_drone._get_sn(timeout=10)
    #for sn in drone_ip_list:
    #    print("scan result: sn:{0}, ip:{1}".format(sn, drone_ip_list[sn]))

    # scan result: sn:0TQZK1FCNT1NFT, ip:('192.168.1.104', 8889)
    #sn_list = list(drone_ip_list.keys())

    #可以通过手动输入来准确控制想要操控的无人机
    #robot_sn_list = [sn_list[0],sn_list[1]]
    robot_sn_list = ["0TQZK1FCNT1NAT", "0TQZK1FCNT1NFT"]
    
    multi_drone.number_id_by_sn([1, robot_sn_list[0]], [2, robot_sn_list[1]])
    multi_drone_group1 = multi_drone.build_group([1, 2])
    multi_drone.run([multi_drone_group1, base_action_1,[20]])
    multi_drone.close()
