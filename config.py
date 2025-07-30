

class Config:
    def __init__(self):
        self.reset()

    def reset(self):
        # 眨眼相关
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 3
        self.COUNTER = 0  # 眨眼帧计数器
        self.TOTAL = 0  # 眨眼总数

        # 哈欠相关
        self.MAR_THRESH = 0.5
        self.MOUTH_AR_CONSEC_FRAMES = 5
        self.mCOUNTER = 0  # 打哈欠帧计数器
        self.mTOTAL = 0  # 打哈欠总数

        # 分心行为
        self.ActionCOUNTER = 0  # 分心行为计数器
        self.hCOUNTER = 0
        self.hTOTAL = 0

        # 疲劳判断变量
        self.Roll = 0  # 整个循环内的帧技术    50帧
        self.Rolleye = 0  # 循环内闭眼帧数
        self.Rollmouth = 0  # 循环内打哈欠数
        self.Rollnod = 0  # 循环内点头次数

        # 2分钟统计
        self.Roll2 = 0  # 2分钟之内的
        self.eye2 = 0
        self.mouth2 = 0
        self.nod2 = 0





config = Config()

