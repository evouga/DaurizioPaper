#include <igl/opengl/glfw/Viewer.h>
#include <igl/triangle/triangulate.h>
#include <Eigen/Sparse>
#include <vector>
#include <igl/boundary_loop.h>
#include <Eigen/Dense>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/cotmatrix.h>

double elasticEnergy(const Eigen::VectorXd &q, const Eigen::MatrixXi &F, const Eigen::MatrixXd &origV,     
    Eigen::VectorXd &deriv, std::vector<Eigen::Triplet<double> > &hess)
{
    int nverts = origV.rows();
    double result = 0;
    deriv.resize(3 * nverts);
    deriv.setZero();
    hess.clear();

    double kstretch = 500.0;
    int nfaces = F.rows();
    for (int i = 0; i < nfaces; i++)
    {
        Eigen::Vector3i face = F.row(i);
        for (int v1 = 0; v1 < 3; v1++)
        {
            int v2 = (v1 + 1) % 3;

            Eigen::Vector3d op1 = origV.row(face[v1]);
            Eigen::Vector3d op2 = origV.row(face[v2]);
            double odist = (op1 - op2).norm();

            Eigen::Vector3d p1 = q.segment<3>(3 * face[v1]);
            Eigen::Vector3d p2 = q.segment<3>(3 * face[v2]);
            double dist = (p1 - p2).norm();
            if (dist < odist)
               continue;
            result += 0.5 * kstretch * (dist - odist)*(dist - odist) / odist / odist;
            deriv.segment<3>(3 * face[v1]) += kstretch * (dist - odist) * (p1 - p2) / dist / odist / odist;
            deriv.segment<3>(3 * face[v2]) -= kstretch * (dist - odist) * (p1 - p2) / dist / odist / odist;
            Eigen::Matrix3d I;
            I.setIdentity();
            Eigen::Matrix3d localH = kstretch*(p1 - p2)*(p1 - p2).transpose() / dist / dist / odist / odist  + kstretch * (dist - odist) * (I / dist - (p1 - p2)*(p1 - p2).transpose() / dist / dist / dist) / odist / odist;

            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                {
                    hess.push_back(Eigen::Triplet<double>(3 * face[v1] + j, 3 * face[v1] + k, localH(j, k)));
                    hess.push_back(Eigen::Triplet<double>(3 * face[v1] + j, 3 * face[v2] + k, -localH(j, k)));
                    hess.push_back(Eigen::Triplet<double>(3 * face[v2] + j, 3 * face[v1] + k, -localH(j, k)));
                    hess.push_back(Eigen::Triplet<double>(3 * face[v2] + j, 3 * face[v2] + k, localH(j, k)));
                }
        }
    }

    double pressure = 1e-3;
    for (int i = 0; i < nfaces; i++)
    {
        Eigen::Vector3d v0 = q.segment<3>(3 * F(i, 0));
        Eigen::Vector3d v1 = q.segment<3>(3 * F(i, 1));
        Eigen::Vector3d v2 = q.segment<3>(3 * F(i, 2));
        result -= pressure / 6.0 * (v0.cross(v1).dot(v2));
        Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
        for (int j = 0; j < 3; j++)
        {
            deriv.segment<3>(3 * F(i, j)) -= pressure / 6.0 * n;
        }
    }

    return result;
}

void createMesh(double w, double h, double triarea, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{
    Eigen::MatrixXd rectV(4, 2);
    Eigen::MatrixXi rectE(4, 2);
    Eigen::MatrixXi H(0, 2);
    

    rectV << -w / 2, -h / 2,
        w / 2, -h / 2,
        w / 2, h / 2,
        -w / 2, h / 2;

    rectE << 0, 1,
        1, 2,
        2, 3,
        3, 0;

    std::stringstream ss;
    ss << "a" << triarea << "q";

    // Triangulate the interior
    igl::triangle::triangulate(rectV, rectE, H, ss.str().c_str(), V, F);

    int nverts = V.rows();
    V.conservativeResize(nverts, 3);
    V.col(2).setZero();
}

double computeVolume(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
{
    double result = 0.0;
    int nfaces = F.rows();
    for (int i = 0; i < nfaces; i++)
    {
        Eigen::Vector3d v0 = V.row(F(i, 0)).transpose();
        Eigen::Vector3d v1 = V.row(F(i, 1)).transpose();
        Eigen::Vector3d v2 = V.row(F(i, 2)).transpose();
        result += 1.0 / 6.0 * (v0.cross(v1)).dot(v2);
    }
    return result;
}

void takeOneStep(Eigen::MatrixXd &curV, const Eigen::MatrixXd &origV, const Eigen::MatrixXi &F, std::set<int> clamped,     
    double &reg)
{
    int nverts = curV.rows();
    int freeDOFs = 3 * nverts - clamped.size();
    std::vector<Eigen::Triplet<double> > proj;
    int row = 0;
    for (int i = 0; i < nverts; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            proj.push_back(Eigen::Triplet<double>(row, 3 * i + j, 1.0));
            row++;
        }
        if (!clamped.count(i))
        {
            proj.push_back(Eigen::Triplet<double>(row, 3 * i + 2, 1.0));
            row++;
        }
    }
    Eigen::SparseMatrix<double> projM(freeDOFs, 3 * nverts);
    projM.setFromTriplets(proj.begin(), proj.end());

    while (true)
    {
        Eigen::VectorXd q(3 * nverts);
        for (int i = 0; i < nverts; i++)
        {
            q.segment<3>(3 * i) = curV.row(i).transpose();
        }
        Eigen::VectorXd derivative;
        std::vector<Eigen::Triplet<double> > hessian;

        double energy = elasticEnergy(q, F, origV, derivative, hessian);        

        Eigen::SparseMatrix<double> H(3 * nverts, 3 * nverts);
        H.setFromTriplets(hessian.begin(), hessian.end());
        Eigen::VectorXd reducedDeriv = projM * derivative;
        Eigen::SparseMatrix<double> reducedH = projM * H * projM.transpose();
        Eigen::SparseMatrix<double> I(freeDOFs, freeDOFs);
        I.setIdentity();
        reducedH += reg * I;        

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver(reducedH);
        Eigen::VectorXd descentDir = solver.solve(-reducedDeriv);
        std::cout << "Solver residual: " << (reducedH*descentDir + reducedDeriv).norm() << std::endl;
        Eigen::VectorXd fullDir = projM.transpose() * descentDir;

        Eigen::MatrixXd newPos = curV;
        for (int i = 0; i < nverts; i++)
        {
            newPos.row(i) += fullDir.segment<3>(3 * i);
        }    
        Eigen::VectorXd newq = q + fullDir;
        double newenergy = elasticEnergy(newq, F, origV, derivative, hessian);        

        if (newenergy <= energy)
        {
            std::cout << "Old energy: " << energy << " new energy: " << newenergy << std::endl;
            std::cout << "Volume is " << computeVolume(newPos, F) << std::endl;
            curV = newPos;
            reg = std::max(1e-6, reg/2.0);
            break;
        }
        else
        {
            reg *= 2.0;
            std::cout << "Old energy: " << energy << " new energy: " << newenergy << " lambda now: " << reg << std::endl;
        }
    }
}

Eigen::MatrixXd origV;
Eigen::MatrixXd curV;
Eigen::MatrixXi F;
std::set<int> clamped;

void reset()
{
    curV = origV;
}

void repaint(igl::opengl::glfw::Viewer &viewer)
{
    viewer.data().clear();
    viewer.data().set_mesh(curV, F);
}

int main(int argc, char *argv[])
{
    createMesh(210, 297, 5.0, origV, F);
    curV = origV;
    std::vector<std::vector<int> > bdry;
    igl::boundary_loop(F, bdry);
    assert(bdry.size() == 1);
    for (int i : bdry[0])
        clamped.insert(i);

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(origV, F);
    viewer.data().set_face_based(true);

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        menu.draw_viewer_menu();

        // Add new group
        if (ImGui::CollapsingHeader("Optimization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Reset", ImVec2(-1, 0)))
            {
                reset();
                repaint(viewer);
            }
            if (ImGui::Button("Optimize Some Step", ImVec2(-1, 0)))
            {
                double reg = 1e-6;
                for (int i = 0; i < 100; i++)
                {
                    takeOneStep(curV, origV, F, clamped, reg);
                }
                repaint(viewer);
            }
            if (ImGui::Button("Save Meshes", ImVec2(-1, 0)))
            {
                igl::writeOBJ("rect.obj", origV, F);
                igl::writeOBJ("solved.obj", curV, F);
            }
        }
    };

    viewer.launch();
}
