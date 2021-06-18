sys.path.append('nx100_remote/') # add relative path
import numpy as np

def get_position():
    return Commands.read_current_specified_coordinate_system_position(str(COORDINATE_SYSTEM), '0')


c_pos = get_position()

Commands.write_linear_move(MoveL.MoveL(
                0, int(self.SPEED), self.COORDINATE_SYSTEM,
                c_pos[0],
                c_pos[1],
                c_pos[2], 
                c_pos[3],
                c_pos[4],
                c_pos[5], 
                Utils.binary_to_decimal(0x00000001)
            ))