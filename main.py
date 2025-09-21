import wx
from wx import glcanvas
from OpenGL.GL import *
import math

# мат часть преобразований
def mat_mul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def mat_vec(m, v):
    return [sum(m[i][j]*v[j] for j in range(4)) for i in range(4)]

def identity():
    return [[1 if i==j else 0 for j in range(4)] for i in range(4)]

def transpose(dx, dy, dz):
    m = identity()
    m[0][3], m[1][3], m[2][3] = dx, dy, dz
    return m

def scale(sx, sy, sz):
    m = identity()
    m[0][0], m[1][1], m[2][2] = sx, sy, sz
    return m

def rot_x(angle):
    m = identity()
    m[1][1], m[1][2] = math.cos(angle), -math.sin(angle)
    m[2][1], m[2][2] = math.sin(angle), math.cos(angle)
    return m

def rot_y(angle):
    m = identity()
    m[0][0], m[0][2] = math.cos(angle), math.sin(angle)
    m[2][0], m[2][2] = -math.sin(angle), math.cos(angle)
    return m

def rot_z(angle):
    m = identity()
    m[0][0], m[0][1] = math.cos(angle), -math.sin(angle)
    m[1][0], m[1][1] = math.sin(angle), math.cos(angle)
    return m

def rot_axis(axis, angle):
    x,y,z = axis
    length = math.sqrt(x*x + y*y + z*z)
    if length == 0: return identity()
    x,y,z = x/length, y/length, z/length
    c, s = math.cos(angle), math.sin(angle)
    mc = 1-c
    m = identity()
    m[0][0] = c + x*x*mc
    m[0][1] = x*y*mc - z*s
    m[0][2] = x*z*mc + y*s
    m[1][0] = y*x*mc + z*s
    m[1][1] = c + y*y*mc
    m[1][2] = y*z*mc - x*s
    m[2][0] = z*x*mc - y*s
    m[2][1] = z*y*mc + x*s
    m[2][2] = c + z*z*mc
    return m

# строим букву P
def define_P(depth=0.3, steps=20):
    height, width = 1.0, 0.3
    radius, cx, cy = height/4, -width, height/4
    path = [(-width, -height/2), (-width, height/2)]
    for i in range(steps+1):
        ang = math.pi/2 - math.pi*i/steps
        x = cx + radius*math.cos(ang)
        y = cy + radius*math.sin(ang)
        path.append((x,y))

    verts = [(x,y, depth/2) for x,y in path] + [(x,y,-depth/2) for x,y in path]
    edges = []
    n = len(path)
    for i in range(n-1): edges.append((i,i+1))
    for i in range(n-1): edges.append((i+n,i+1+n))
    for i in range(n): edges.append((i,i+n))
    return verts, edges

class Canvas(glcanvas.GLCanvas):
    def __init__(self, parent):
        super().__init__(parent, -1, attribList=[glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER])
        self.ctx = glcanvas.GLContext(self)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda e: self.Refresh(False), self.timer)
        self.timer.Start(16)

        # трансформации буквы
        self.rx=self.ry=self.rz=0.0
        self.scale_factor = 1.0
        self.tx,self.ty,self.tz = 0,0,3
        self.extra_rotations = []

        # геометрия буквы
        self.verts,self.edges = define_P()

        # оси
        self.axes = [((1,0,0),(1,0,0),"X"), ((0,1,0),(0,.6,0),"Y"),
                     ((0,0,1),(0,0,1),"Z"), ((1,1,0),(.6,0,.8),"(1,1,0)"),
                     ((1,1,1),(.2,.2,.2),"(1,1,1)")]
        self.show_axes = True

        self.Bind(wx.EVT_PAINT, self.paint)
        self.Bind(wx.EVT_SIZE, self.resize)

    def resize(self,e):
        w,h = self.GetClientSize()
        self.SetCurrent(self.ctx)
        glViewport(0,0,w,h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0,w,0,h,-1,1)
        glMatrixMode(GL_MODELVIEW)

    # простая 3D -> 2D проекция с фиксированной камерой
    def proj(self, m, v, w, h, c=5.0, eps=1e-6):
        xw, yw, zw, _ = mat_vec(m, [v[0],v[1],v[2],1])
        d = c - zw
        if d <= eps: return None
        t = c / d
        return (w/2 + t*xw*w/3, h/2 + t*yw*h/3)

    # вывод текста
    def txt(self, x, y, s):
        glColor3f(0,0,0)
        glRasterPos2f(x,y)
        for ch in s:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))

    # отрисовка
    def paint(self, e):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.ctx)
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        w, h = self.GetClientSize()

        # буква
        m_letter = scale(self.scale_factor, self.scale_factor, self.scale_factor)
        m_letter = mat_mul(rot_x(self.rx), m_letter)
        m_letter = mat_mul(rot_y(self.ry), m_letter)
        m_letter = mat_mul(rot_z(self.rz), m_letter)
        for axis, ang in self.extra_rotations:
            m_letter = mat_mul(rot_axis(axis, ang), m_letter)
        m_letter = mat_mul(transpose(self.tx, self.ty, self.tz), m_letter)

        glColor3f(.1, .2, .8)
        glBegin(GL_LINES)
        for i, j in self.edges:
            p1 = self.proj(m_letter, self.verts[i], w, h)
            p2 = self.proj(m_letter, self.verts[j], w, h)
            if p1 and p2: glVertex2f(*p1); glVertex2f(*p2)
        glEnd()

        # оси мира
        if self.show_axes:
            origin = (0, 0, 0)
            axis_scale = 1.0
            offset = 0.1

            for vec, col, label in self.axes:
                if label in ("(1,1,0)", "(1,1,1)"):
                    end3d = (vec[0] * axis_scale + offset, vec[1] * axis_scale + offset, vec[2] * axis_scale + offset)
                else:
                    end3d = (vec[0] * axis_scale, vec[1] * axis_scale, vec[2] * axis_scale)

                p1 = self.proj(identity(), origin, w, h)
                p2 = self.proj(identity(), end3d, w, h)
                if not p1 or not p2: continue

                glColor3f(*col)
                glBegin(GL_LINES)
                glVertex2f(*p1)
                glVertex2f(*p2)
                glEnd()
                self.txt(p2[0] + 5, p2[1] + 5, label)

        self.SwapBuffers()


class Frame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Letter P", size=(900,700))
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.cv = Canvas(panel)
        sizer.Add(self.cv,1,wx.EXPAND)

        grid = wx.GridSizer(6,6,5,5)
        def btn(label, fn): b=wx.Button(panel,label=label); b.Bind(wx.EVT_BUTTON,fn); grid.Add(b,0,wx.EXPAND)

        # управление
        btn("X+", lambda e:self.move(.1,0,0)); btn("X-", lambda e:self.move(-.1,0,0))
        btn("Y+", lambda e:self.move(0,.1,0)); btn("Y-", lambda e:self.move(0,-.1,0))
        btn("Z+", lambda e:self.move(0,0,.1)); btn("Z-", lambda e:self.move(0,0,-.1))
        btn("RotX+", lambda e:self.rot(.1,0,0)); btn("RotX-", lambda e:self.rot(-.1,0,0))
        btn("RotY+", lambda e:self.rot(0,.1,0)); btn("RotY-", lambda e:self.rot(0,-.1,0))
        btn("RotZ+", lambda e:self.rot(0,0,.1)); btn("RotZ-", lambda e:self.rot(0,0,-.1))
        btn("Axis(1,1,0)+", lambda e:self.rot_axis((1,1,0),.1))
        btn("Axis(1,1,0)-", lambda e:self.rot_axis((1,1,0),-.1))
        btn("Axis(1,1,1)+", lambda e:self.rot_axis((1,1,1),.1))
        btn("Axis(1,1,1)-", lambda e:self.rot_axis((1,1,1),-.1))
        btn("Scale+", lambda e:self.scale(1.1))
        btn("Scale-", lambda e:self.scale(.9))
        btn("Toggle Axes", lambda e:self.toggle_axes())

        sizer.Add(grid,0,wx.ALL|wx.EXPAND,10)
        panel.SetSizer(sizer)
        self.Show()

    # управление из кнопок
    def move(self, dx, dy, dz): self.cv.tx+=dx; self.cv.ty+=dy; self.cv.tz+=dz
    def rot(self, rx, ry, rz): self.cv.rx+=rx; self.cv.ry+=ry; self.cv.rz+=rz
    def rot_axis(self, axis, ang): self.cv.extra_rotations.append((axis, ang))
    def scale(self, f): self.cv.scale_factor *= f
    def toggle_axes(self): self.cv.show_axes = not self.cv.show_axes

if __name__=="__main__":
    from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_12
    glutInit() # для вывода текста
    app = wx.App(False)
    Frame()
    app.MainLoop()
