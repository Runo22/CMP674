#include "InteropTypes.cuh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static const int kWindowWidth = 1280;
static const int kWindowHeight = 720;
static int g_clustered = 0;
static int g_paused = 0;

static GLuint compile_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        std::fprintf(stderr, "Shader compile failed: %s\n", log);
        std::exit(EXIT_FAILURE);
    }

    return shader;
}

static GLuint create_shader_program(void) {
    const char* vertex_source =
        "#version 330 core\n"
        "layout(location = 0) in vec2 aPos;\n"
        "layout(location = 1) in vec3 aColor;\n"
        "layout(location = 2) in float aSize;\n"
        "out vec3 vColor;\n"
        "void main() {\n"
        "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
        "    gl_PointSize = aSize;\n"
        "    vColor = aColor;\n"
        "}\n";

    const char* fragment_source =
        "#version 330 core\n"
        "in vec3 vColor;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "    vec2 p = gl_PointCoord * 2.0 - 1.0;\n"
        "    float d = dot(p, p);\n"
        "    if (d > 1.0) discard;\n"
        "    float shade = 1.0 - d * 0.35;\n"
        "    FragColor = vec4(vColor * shade, 1.0);\n"
        "}\n";

    GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        std::fprintf(stderr, "Program link failed: %s\n", log);
        std::exit(EXIT_FAILURE);
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return program;
}

static void key_callback(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) {
        return;
    }

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_SPACE) {
        g_paused = !g_paused;
    } else if (key == GLFW_KEY_C) {
        g_clustered = !g_clustered;
        cuda_visualizer_reset(g_clustered);
    } else if (key == GLFW_KEY_R) {
        cuda_visualizer_reset(g_clustered);
    }
}

int main(int argc, char** argv) {
    int object_count = 2500;
    if (argc > 1) {
        object_count = std::atoi(argv[1]);
        if (object_count < 100) {
            object_count = 100;
        }
    }

    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW.\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        kWindowWidth,
        kWindowHeight,
        "CUDA OpenGL Collision Visualizer",
        NULL,
        NULL);
    if (window == NULL) {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, key_callback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::fprintf(stderr, "Failed to initialize GLEW.\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint program = create_shader_program();
    GLuint vao = 0;
    GLuint vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        (GLsizeiptr)object_count * (GLsizeiptr)sizeof(RenderVertex),
        NULL,
        GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    if (!cuda_visualizer_create(vbo, object_count, kWindowWidth, kWindowHeight)) {
        std::fprintf(stderr, "Failed to create CUDA visualizer.\n");
        return 1;
    }

    double last_time = glfwGetTime();
    VisualizerMetrics metrics;
    std::memset(&metrics, 0, sizeof(metrics));

    while (!glfwWindowShouldClose(window)) {
        const double now = glfwGetTime();
        float dt = (float)(now - last_time);
        last_time = now;
        if (dt > 0.033f) {
            dt = 0.033f;
        }

        if (!g_paused) {
            cuda_visualizer_step(dt, &metrics);
        }

        char title[512];
        std::snprintf(
            title,
            sizeof(title),
            "CUDA-OpenGL Interop | objects=%d | distribution=%s | collisions=%llu | candidates=%llu | gpu frame=%.3f ms | Space pause | C cluster | R reset",
            object_count,
            g_clustered ? "clustered" : "uniform",
            metrics.collision_count,
            metrics.candidate_pair_count,
            metrics.gpu_time_ms);
        glfwSetWindowTitle(window, title);

        glViewport(0, 0, kWindowWidth, kWindowHeight);
        glClearColor(0.035f, 0.040f, 0.050f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, object_count);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cuda_visualizer_destroy();
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
