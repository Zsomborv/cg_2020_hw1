//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Varga Zsombor
// Neptun : AKCJOP
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders


const int nv = 1000;
std::vector<vec2> triangleVertices;
std::vector<vec2> triangleTemp;

class Points {
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	std::vector<vec2> wCtrlPoints;

public:
	void Create() {
		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);

		glGenBuffers(1, &vboCtrlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
	}

	void AddControlPoint(float cX, float cY) {
		if (1 > sqrt((pow(cX, 2) + pow(cY, 2)))) {
			if (wCtrlPoints.size() == 3) {
				wCtrlPoints.clear();
			}
			wCtrlPoints.push_back(vec2(cX, cY));
		}
	}

	void Draw() {
		if (wCtrlPoints.size() > 0) {
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * sizeof(vec2), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(1, 0, 0), "color");
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}
	}

	int getSize() {
		return wCtrlPoints.size();
	}

	vec2 getCtrlPoints(int idx) {
		return wCtrlPoints[idx];
	}
};

class Circle {
    unsigned int vao, vao2;
    float cx, cy;
    float r;
    float re, g, b;
    vec2 cp1, cp2;
    int nV;
    
public:

    vec3 Calculate(vec2 p1, vec2 p2) {
        cp1 = p1;
        cp2 = p2;
        cy = (1 + pow(p1.x, 2) + pow(p1.y, 2) - ((p1.x * (pow(p1.x, 2) - pow(p2.x, 2))) / (p1.x - p2.x)) - ((p1.x * (pow(p1.y, 2) - pow(p2.y, 2))) / (p1.x - p2.x))) / ((2 * p1.y) - (2 * ((p1.x * (p1.y - p2.y)) / (p1.x - p2.x))));
        cx = (pow(p1.x, 2) - pow(p2.x, 2) - 2 * cy * (p1.y - p2.y) + pow(p1.y, 2) - pow(p2.y, 2)) / (2 * (p1.x - p2.x));
        r = sqrt(pow((p1.x - cx), 2) + pow((p1.y - cy), 2));

        return vec3(cx, cy, r);
    }

    vec2 getXY() {
        return vec2(cx, cy);
    }

    float getRadius() {
        return r;
    }

    void Create(float cr, float cg, float cb) {
        r = 1.0f;
        re = cr;
        g = cg;
        b = cb;

        glGenVertexArrays(1, &vao);		// get 1 vao id
        glBindVertexArray(vao);			// make it active

        unsigned int vbo;				// vertex buffer object
        glGenBuffers(1, &vbo);			// Generate 1 buffer

        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        vec2 vertices[nv];
        for (int i = 0; i < nv; i++)
        {
            float fi = i * 2 * M_PI / nv;
            vertices[i] = vec2(cosf(fi), sinf(fi));
        }
        glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
            sizeof(vec2) * nv,			// # bytes
            vertices,					// address
            GL_STATIC_DRAW				// we do not change later
            );
        glEnableVertexAttribArray(0);	// AttribArray 0
        glVertexAttribPointer(0,		// vbo -> AttribArray 0
            2, GL_FLOAT, GL_FALSE,		// two floats/attrib, not fixed-point
            0, NULL);					// stride, offset: tightly packed
    }

    void Draw() {
        if (re == 1.0f && b == 1.0f && g == 1.0f) {
            r = 1;
            cy = 0;
            cx = 0;
        }

        // Set color
        int location = glGetUniformLocation(gpuProgram.getId(), "color");
        glUniform3f(location, re, g, b); // 3 floats

        float MVPtransf[4][4] = { r, 0, 0, 0,    // MVP matrix, sizex
                                  0, r, 0, 0,    // row-major! sizey
                                  0, 0, 1, 0,
                                  cx, cy, 0, 1 };

        location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
        glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

        glBindVertexArray(vao);  // Draw call

        glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, nv /*# Elements*/);
    }

    void CreateArc() {
        glGenVertexArrays(1, &vao2);		// get 1 vao id
        glBindVertexArray(vao2);			// make it active

        unsigned int vbo;				// vertex buffer object
        glGenBuffers(1, &vbo);			// Generate 1 buffer

        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        float firstAngle = atan2(cp1.y - cy, cp1.x - cx);
        float lastAngle = atan2(cp2.y - cy, cp2.x - cx);

        int invertedflag = 0;
        float deltaAngle = lastAngle - firstAngle;

        if (deltaAngle > M_PI || (deltaAngle < 0 && deltaAngle > -M_PI)) {
            float temp = lastAngle;
            lastAngle = firstAngle;
            firstAngle = temp;
            invertedflag = 1;
        }
        if (lastAngle < 0 && firstAngle > 0)
            lastAngle = 2 * M_PI + lastAngle;

        int i0 = int(firstAngle * nv / (2 * M_PI));
        int i1 = int(lastAngle * nv / (2 * M_PI));

        std::vector<vec2> vertices(i1 - i0 + 3);
        int k = 0;

        vertices[0] = vec2(cosf(firstAngle), sinf(firstAngle));
        for (int i = i0; i < i1; i++) {
            float fi = i * 2 * M_PI / nv;
            if (fi < lastAngle && fi > firstAngle) {
                vertices[++k] = vec2(cosf(fi), sinf(fi));
            }
        }

        vertices[++k] = vec2(cosf(lastAngle), sinf(lastAngle));

       if (invertedflag)
            for (int i = k; i > 0; i--)
            {
                vec2 t = vec2(vertices[i].x * r + cx, vertices[i].y * r + cy);
                triangleVertices.push_back(t);
              
            }
        else
            for (int i = 0; i < k; i++)
            {
                vec2 t = vec2(vertices[i].x * r + cx, vertices[i].y * r + cy);
                triangleVertices.push_back(t);
                
            }

        nV = k + 1;
                
        glBufferData(GL_ARRAY_BUFFER,	// Copy to GPU target
            sizeof(vec2) * nV,			// # bytes
            &vertices[0],					// address
            GL_STATIC_DRAW				// we do not change later
            );
        
        glEnableVertexAttribArray(0);	// AttribArray 0
        glVertexAttribPointer(0,		// vbo -> AttribArray 0
            2, GL_FLOAT, GL_FALSE,		// two floats/attrib, not fixed-point
            0, NULL);					// stride, offset: tightly packed
         
    }

    void DrawArc() {

        // Set color
        int location = glGetUniformLocation(gpuProgram.getId(), "color");
        glUniform3f(location, 0.0f, 0.0f, 1.0f); // 3 floats

        float MVPtransf[4][4] = { r, 0, 0, 0,    // MVP matrix, sizex
                                  0, r, 0, 0,    // row-major! sizey
                                  0, 0, 1, 0,
                                  cx, cy, 0, 1 };

        location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
        glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

        glBindVertexArray(vao2);  // Draw call
        glDrawArrays(GL_LINE_STRIP, 0 /*startIdx*/, nV /*# Elements*/);
    }

};

class Triangle {
    unsigned int vaoT;
    float triangle[6];
    float red, green, blue;
public:

    void CreateTriangle(vec2 p1, vec2 p2, vec2 p3, float r, float g, float b) {
        red = r;
        green = g;
        blue = b;

        glGenVertexArrays(1, &vaoT);	// get 1 vao id
        glBindVertexArray(vaoT);		// make it active

        unsigned int vbo;		// vertex buffer object
        glGenBuffers(1, &vbo);	// Generate 1 buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
       
        triangle[0] = p1.x;
        triangle[1] = p1.y;
        triangle[2] = p2.x;
        triangle[3] = p2.y;
        triangle[4] = p3.x;
        triangle[5] = p3.y; // Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

        glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
            sizeof(vec2)*3,  // # bytes
            triangle,	      	// address
            GL_STATIC_DRAW);	// we do not change later

        glEnableVertexAttribArray(0);  // AttribArray 0
        glVertexAttribPointer(0,       // vbo -> AttribArray 0
            2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
            0, NULL); 		     // stride, offset: tightly packed
    }

    void DrawTriangle(){
        int location = glGetUniformLocation(gpuProgram.getId(), "color");
        glUniform3f(location, red, green, blue); // 3 floats

        float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
                                  0, 1, 0, 0,    // row-major!
                                  0, 0, 1, 0,
                                  0, 0, 0, 1 };

        location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
        glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

        glBindVertexArray(vaoT);  // Draw call
        glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 /*# Elements*/);
    }
};

std::vector<Triangle> triangleColoring;

void calcSides(vec2 p1, vec2 p2, vec2 p3) {

    float sideA = 0;
    float sideB = 0;
    float sideC = 0;

    int idxA=0, idxB=0, idxC=0;

    for (int k = 0; k < triangleVertices.size(); k++)
    {
        if (triangleVertices.at(k).x < p1.x + 0.00001f && triangleVertices.at(k).x > p1.x - 0.00001f) {
            idxA = k;
        }
        if (triangleVertices.at(k).x < p2.x + 0.00001f && triangleVertices.at(k).x > p2.x - 0.00001f) {
            idxB = k;      
        }
        if (triangleVertices.at(k).x < p3.x+ 0.00001f && triangleVertices.at(k).x > p3.x - 0.00001f) {
            idxC = k;
        }
            
    }
    //Side A
    int i = 0;
    while (i < idxB) {
        sideA += sqrt(pow(triangleVertices.at(i + 1).x - triangleVertices.at(i).x, 2) +
            pow(triangleVertices.at(i + 1).y - triangleVertices.at(i).y, 2)) /
            (1 - pow(triangleVertices.at(i).x, 2) - pow(triangleVertices.at(i).y, 2));
        i++;
    }
    
    //Side B
    i--;
    while (i <=idxC) {
        sideB += sqrt(pow(triangleVertices.at(i).x - triangleVertices.at(i-1).x, 2) +
            pow(triangleVertices.at(i).y - triangleVertices.at(i-1).y, 2)) /
            (1 - pow(triangleVertices.at(i-1).x, 2) - pow(triangleVertices.at(i-1).y, 2));
        i++;
    }
 
    //Side C
    i--;
    while (i < triangleVertices.size()-1) {
        sideC += sqrt(pow(triangleVertices.at(i + 1).x - triangleVertices.at(i).x, 2) +
            pow(triangleVertices.at(i + 1).y - triangleVertices.at(i).y, 2)) /
            (1 - pow(triangleVertices.at(i).x, 2) - pow(triangleVertices.at(i).y, 2));
        i++;
    }
   
    printf("a: %f,\tb: %f,\tc: %f\n", sideA, sideB, sideC);
 }


Circle baseCircle;
Circle circle1;
Circle circle2;
Circle circle3;
Points points;

void calcDeg(vec2 p1, vec2 p2, vec2 p3) {

    //ControlCalc
    int idxA = 0, idxB = 0, idxC = 0;
    float delta;

    for (int k = 0; k < triangleVertices.size(); k++)
    {
        if (triangleVertices.at(k).x < p1.x + 0.00001f && triangleVertices.at(k).x > p1.x - 0.00001f) {
            idxA = k;
        }
        if (triangleVertices.at(k).x < p2.x + 0.00001f && triangleVertices.at(k).x > p2.x - 0.00001f) {
            idxB = k;
        }
        if (triangleVertices.at(k).x < p3.x + 0.00001f && triangleVertices.at(k).x > p3.x - 0.00001f) {
            idxC = k;
        }

    }
    float firstControlAngle;
    float lastControlAngle;

    //1
    firstControlAngle = atan2(triangleVertices.at(idxB - 1).x - p2.x, triangleVertices.at(idxB - 1).y - p2.y);
    lastControlAngle = atan2(triangleVertices.at(idxB + 1).x - p2.x, triangleVertices.at(idxB + 1).y - p2.y);

    if (lastControlAngle - firstControlAngle > M_PI || (lastControlAngle - firstControlAngle < 0 && lastControlAngle - firstControlAngle > -M_PI)) {
        float temp = lastControlAngle;
        lastControlAngle = firstControlAngle;
        firstControlAngle = temp;
    }
    if (lastControlAngle < 0 && firstControlAngle > 0)
        lastControlAngle = 2 * M_PI + lastControlAngle;

    delta = lastControlAngle - firstControlAngle;
 
    //ALPHA
    float firstAngleAlpha = atan2(circle1.getXY().x - p2.x, circle1.getXY().y - p2.y);
    float lastAngleAlpha = atan2(circle2.getXY().x - p2.x, circle2.getXY().y - p2.y);

    if (lastAngleAlpha - firstAngleAlpha > M_PI || (lastAngleAlpha - firstAngleAlpha < 0 && lastAngleAlpha - firstAngleAlpha > -M_PI)) {
        float temp = lastAngleAlpha;
        lastAngleAlpha = firstAngleAlpha;
        firstAngleAlpha = temp;
    }
    if (lastAngleAlpha < 0 && firstAngleAlpha > 0)
        lastAngleAlpha = 2 * M_PI + lastAngleAlpha;
 
    float alpha = lastAngleAlpha - firstAngleAlpha;
    alpha = M_PI - alpha;
    if (delta < M_PI / 2) {
        if (alpha > M_PI/2) {
            alpha = M_PI - alpha;
        }
    }

    //2
    firstControlAngle = atan2(triangleVertices.at(idxC - 1).x - p3.x, triangleVertices.at(idxC - 1).y - p3.y);
    lastControlAngle = atan2(triangleVertices.at(idxC + 1).x - p3.x, triangleVertices.at(idxC + 1).y - p3.y);

    if (lastControlAngle - firstControlAngle > M_PI || (lastControlAngle - firstControlAngle < 0 && lastControlAngle - firstControlAngle > -M_PI)) {
        float temp = lastControlAngle;
        lastControlAngle = firstControlAngle;
        firstControlAngle = temp;
    }
    if (lastControlAngle < 0 && firstControlAngle > 0)
        lastControlAngle = 2 * M_PI + lastControlAngle;

    delta = lastControlAngle - firstControlAngle;

    //BETA
    float firstAngleBeta = atan2(circle2.getXY().x - p3.x, circle2.getXY().y - p3.y);
    float lastAngleBeta = atan2(circle3.getXY().x - p3.x, circle3.getXY().y - p3.y);

    if (lastAngleBeta - firstAngleBeta > M_PI || (lastAngleBeta - firstAngleBeta < 0 && lastAngleBeta - firstAngleBeta > -M_PI)) {
        float temp = lastAngleBeta;
        lastAngleBeta = firstAngleBeta;
        firstAngleBeta = temp;
    }
    if (lastAngleBeta < 0 && firstAngleBeta > 0)
        lastAngleBeta = 2 * M_PI + lastAngleBeta;

    float beta = lastAngleBeta - firstAngleBeta;
    beta = M_PI - beta;
    if (beta > M_PI / 2 && delta < M_PI / 2) {
        beta = M_PI -beta;
    }

    //3
    firstControlAngle = atan2(triangleVertices.at(triangleVertices.size() - 1).x - p1.x, triangleVertices.at(triangleVertices.size() - 1).y - p1.y);
    lastControlAngle = atan2(triangleVertices.at(idxA + 1).x - p1.x, triangleVertices.at(idxA + 1).y - p1.y);

    if (lastControlAngle - firstControlAngle > M_PI || (lastControlAngle - firstControlAngle < 0 && lastControlAngle - firstControlAngle > -M_PI)) {
        float temp = lastControlAngle;
        lastControlAngle = firstControlAngle;
        firstControlAngle = temp;
    }
    if (lastControlAngle < 0 && firstControlAngle > 0)
        lastControlAngle = 2 * M_PI + lastControlAngle;

    delta = lastControlAngle - firstControlAngle;

    //GAMMA    
    float firstAngleGamma = atan2(circle1.getXY().x - p1.x, circle1.getXY().y - p1.y);
    float lastAngleGamma = atan2(circle3.getXY().x - p1.x, circle3.getXY().y - p1.y);

    if (lastAngleGamma - firstAngleGamma > M_PI || (lastAngleGamma - firstAngleGamma < 0 && lastAngleGamma - firstAngleGamma > -M_PI)) {
        float temp = lastAngleGamma;
        lastAngleGamma = firstAngleGamma;
        firstAngleGamma = temp;
    }
    if (lastAngleGamma < 0 && firstAngleGamma > 0)
        lastAngleGamma = 2 * M_PI + lastAngleGamma;

    float gamma = lastAngleGamma - firstAngleGamma;
    gamma = M_PI - gamma;
    if (gamma > M_PI / 2 && delta < M_PI / 2) {
        gamma = M_PI - gamma;
    }
  
    printf("alpha: %f,\tbeta: %f,\tgamma: %f,\tAngle sum: %f\n", alpha * 180 / M_PI, beta * 180 / M_PI, gamma * 180 / M_PI, alpha * 180 / M_PI + beta * 180 / M_PI + gamma * 180 / M_PI);

}

bool Konvex(vec2 mid, vec2 p, int idx1, int idx2) {
    float mpd = sqrt(pow(mid.x - p.x, 2) + pow(mid.y - p.y, 2));
    
    vec2 tmp = triangleVertices.at((idx1 + idx2) / 2);

    float pcpd = sqrt(pow(tmp.x - p.x, 2) + pow(tmp.y - p.y, 2));

    if (mpd > pcpd -0.001f) {
       return false;
    }
    return true;

}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);

	baseCircle.Create(1.0f, 1.0f, 1.0f);
	points.Create();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	baseCircle.Draw();
	points.Draw();

	if (points.getSize() > 2) {
       
        vec2 p1 = points.getCtrlPoints(0);
        vec2 p2 = points.getCtrlPoints(1);
        vec2 p3 = points.getCtrlPoints(2);

		vec3 v1 = circle1.Calculate(p1, p2);
		vec3 v2 = circle2.Calculate(p2, p3);
		vec3 v3 = circle3.Calculate(p3, p1);

        circle1.CreateArc();
        circle2.CreateArc();
        circle3.CreateArc();

        for (int i = 0; i < triangleVertices.size(); i++)
        {
            triangleTemp.push_back(triangleVertices.at(i));

        }

        while (triangleTemp.size() > 3) {
            for (int i = 0; i < triangleTemp.size()-2; i++)
            {               
                Triangle tr;
                tr.CreateTriangle(triangleTemp.at(i), triangleTemp.at(i + 1), triangleTemp.at(i + 2), 0.4f, 1.0f, 0.3f);
                triangleColoring.push_back(tr);
                triangleTemp.erase(triangleTemp.begin() + i + 1);
               
            } 
        }
        
        Triangle lasttr;
        lasttr.CreateTriangle(triangleTemp.at(0), triangleTemp.at(1), triangleTemp.at(2), 0.4f, 1.0f, 0.3f);
        triangleColoring.push_back(lasttr);
        
        for (int i = 0; i < triangleColoring.size(); i++)
        {
            triangleColoring.at(i).DrawTriangle();
        }

        float idxA, idxB, idxC;
        std::vector<Triangle> triangleCorrect;
        for (int k = 0; k < triangleVertices.size(); k++)
        {
            if (triangleVertices.at(k).x < p1.x + 0.00001f && triangleVertices.at(k).x > p1.x - 0.00001f) {
                idxA = k;
            }
            if (triangleVertices.at(k).x < p2.x + 0.00001f && triangleVertices.at(k).x > p2.x - 0.00001f) {
                idxB = k;
            }
            if (triangleVertices.at(k).x < p3.x + 0.00001f && triangleVertices.at(k).x > p3.x - 0.00001f) {
                idxC = k;
            }

        }
     
        vec2 mid1 = vec2((p1.x+p2.x)/2, (p1.y + p2.y) / 2);
        vec2 mid2 = vec2((p3.x+p2.x)/2, (p3.y + p2.y) / 2);
        vec2 mid3 = vec2((p1.x+p3.x)/2, (p1.y + p3.y) / 2);

        for (int i = 0; i < idxB; i++) {
            Triangle t;
            t.CreateTriangle(triangleVertices.at(i), triangleVertices.at(i + 1), mid1, 1.0f, 1.0f, 1.0f);
            if(!Konvex(mid1, p3, idxA, idxB))
                triangleCorrect.push_back(t);
        }

        for (int i = idxB; i < idxC; i++) {
            Triangle t;
            t.CreateTriangle(triangleVertices.at(i), triangleVertices.at(i + 1), mid2, 1.0f, 1.0f, 1.0f);
            if (!Konvex(mid2, p1, idxB, idxC))
                triangleCorrect.push_back(t);
        }

        for (int i = idxC; i < triangleVertices.size()-1; i++) {
            Triangle t;
            t.CreateTriangle(triangleVertices.at(i), triangleVertices.at(i + 1), mid3, 1.0f, 1.0f, 1.0f);
            if (!Konvex(mid3, p2, triangleVertices.size()-1, idxC))
            triangleCorrect.push_back(t);
        }

        for (int i = 0; i < triangleCorrect.size(); i++)
        {
            triangleCorrect.at(i).DrawTriangle();
        }

        

        circle1.DrawArc();
        circle2.DrawArc();
        circle3.DrawArc();

       calcDeg(p1, p2, p3);
       calcSides(p1, p2, p3);
       triangleTemp.clear();
       triangleVertices.clear();
       triangleColoring.clear();
	}
   
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		points.AddControlPoint(cX, cY);
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

