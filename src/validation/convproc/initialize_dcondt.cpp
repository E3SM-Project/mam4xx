// mam4xx: Copyright (c) 2022,
// Battelle Memorial Institute and
// National Technology & Engineering Solutions of Sandia, LLC (NTESS)
// SPDX-License-Identifier: BSD-3-Clause

#include <ekat_assert.hpp>

#include <mam4xx/convproc.hpp>
#include <validation.hpp>

using namespace skywalker;

namespace {
void get_input(const Input &input, const std::string &name, const int size,
               std::vector<Real> &host, mam4::ColumnView &dev) {
  host = input.get_array(name);
  EKAT_ASSERT(host.size() == size);
  dev = mam4::validation::create_column_view(size);
  auto host_view = Kokkos::create_mirror_view(dev);
  for (int n = 0; n < size; ++n)
    host_view[n] = host[n];
  Kokkos::deep_copy(dev, host_view);
}
void get_input(const Input &input, const std::string &name, const int rows,
               const int cols, std::vector<Real> &host,
               Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                            Kokkos::MemoryUnmanaged> &dev) {
  host = input.get_array(name);
  EKAT_ASSERT(host.size() == rows * cols);
  mam4::ColumnView col_view = mam4::validation::create_column_view(rows * cols);
  dev = Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                     Kokkos::MemoryUnmanaged>(col_view.data(), rows, cols);
  {
    std::vector<std::vector<Real>> matrix(rows, std::vector<Real>(cols));
    // Col Major layout
    for (int i = 0, n = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j, ++n)
        matrix[i][j] = host[n];
    auto host_view = Kokkos::create_mirror_view(dev);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        host_view(i, j) = matrix[i][j];
    Kokkos::deep_copy(dev, host_view);
  }
}
void set_output(Output &output, const std::string &name, const int rows,
                const int cols, std::vector<Real> &host,
                const Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                                   Kokkos::MemoryUnmanaged> &dev) {
  host.resize(rows * cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0, n = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j, ++n)
      host[n] = host_view(i, j);
  output.set(name, host);
}
void set_host(const std::string &name, const int rows, const int cols,
              std::vector<Real> &host,
              const Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                                 Kokkos::MemoryUnmanaged> &dev) {
  host.resize(rows * cols);
  auto host_view = Kokkos::create_mirror_view(dev);
  Kokkos::deep_copy(host_view, dev);
  for (int i = 0, n = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j, ++n)
      host[n] = host_view(i, j);
}
} // namespace
void initialize_dcondt(Ensemble *ensemble) {
  // We don't need any settings for this particular test.
  // Settings settings = ensemble->settings();
  // Run the ensemble.
  ensemble->process([=](const Input &input, Output &output) {
    const int nlev = 72;
    // Fetch ensemble parameters
    // Convert to C++ index by subtracting one.
    const int ktop = input.get("ktop") - 1;
    EKAT_ASSERT(47 == ktop);
    const int kbot = input.get("kbot");
    EKAT_ASSERT(71 == kbot);
    const int iflux_method = input.get("iflux_method");
    EKAT_ASSERT(1 == iflux_method);

    std::vector<Real> doconvproc_extd_host, dpdry_i_host, fa_u_host, mu_i_host,
        md_i_host, chat_host, gath_host, conu_host, cond_host,
        dconudt_activa_host, dconudt_wetdep_host, dudp_host, dddp_host,
        eudp_host, eddp_host, dcondt_host, dcondt_host_2;
    mam4::ColumnView doconvproc_extd_dev, dpdry_i_dev, fa_u_dev, mu_i_dev,
        md_i_dev, dudp_dev, dddp_dev, eudp_dev, eddp_dev;
    Kokkos::View<Real * [mam4::ConvProc::pcnst_extd], Kokkos::MemoryUnmanaged>
        gath_dev, chat_dev, conu_dev, cond_dev, dconudt_activa_dev,
        dconudt_wetdep_dev, dcondt_dev, dcondt_dev_2;

    get_input(input, "doconvproc_extd", mam4::ConvProc::pcnst_extd,
              doconvproc_extd_host, doconvproc_extd_dev);

    get_input(input, "dpdry_i", nlev, dpdry_i_host, dpdry_i_dev);
    get_input(input, "fa_u", nlev, fa_u_host, fa_u_dev);
    get_input(input, "dudp", nlev, dudp_host, dudp_dev);
    get_input(input, "dddp", nlev, dddp_host, dddp_dev);
    get_input(input, "eudp", nlev, eudp_host, eudp_dev);
    get_input(input, "eddp", nlev, eddp_host, eddp_dev);

    get_input(input, "mu_i", nlev + 1, mu_i_host, mu_i_dev);
    get_input(input, "md_i", nlev + 1, md_i_host, md_i_dev);

    get_input(input, "chat", nlev + 1, mam4::ConvProc::pcnst_extd, chat_host,
              chat_dev);
    get_input(input, "const", nlev, mam4::ConvProc::pcnst_extd, gath_host,
              gath_dev);
    get_input(input, "conu", nlev + 1, mam4::ConvProc::pcnst_extd, conu_host,
              conu_dev);
    get_input(input, "cond", nlev + 1, mam4::ConvProc::pcnst_extd, cond_host,
              cond_dev);

    get_input(input, "dconudt_activa", nlev + 1, mam4::ConvProc::pcnst_extd,
              dconudt_activa_host, dconudt_activa_dev);
    get_input(input, "dconudt_wetdep", nlev + 1, mam4::ConvProc::pcnst_extd,
              dconudt_wetdep_host, dconudt_wetdep_dev);

    mam4::ColumnView col_view =
        mam4::validation::create_column_view(nlev * mam4::ConvProc::pcnst_extd);
    dcondt_dev = Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                              Kokkos::MemoryUnmanaged>(
        col_view.data(), nlev, mam4::ConvProc::pcnst_extd);
    mam4::ColumnView col_view_2 =
        mam4::validation::create_column_view(nlev * mam4::ConvProc::pcnst_extd);
    dcondt_dev_2 = Kokkos::View<Real * [mam4::ConvProc::pcnst_extd],
                                Kokkos::MemoryUnmanaged>(
        col_view_2.data(), nlev, mam4::ConvProc::pcnst_extd);

        auto team_policy = mam4::ThreadTeamPolicy(1u, 1u);
        Kokkos::parallel_for(
           team_policy, KOKKOS_LAMBDA(const mam4::ThreadTeam &team) {
          bool doconvproc_extd[mam4::ConvProc::pcnst_extd];
          Real dpdry_i[nlev];
          Real fa_u[nlev];
          Real mu_i[nlev + 1];
          Real md_i[nlev + 1];
          Real dudp[nlev];
          Real dddp[nlev];
          Real eudp[nlev];
          Real eddp[nlev];

          for (int i = 0; i < mam4::ConvProc::pcnst_extd; ++i)
            doconvproc_extd[i] = doconvproc_extd_dev[i];
          for (int i = 0; i < nlev; ++i)
            dpdry_i[i] = dpdry_i_dev(i);
          for (int i = 0; i < nlev; ++i)
            fa_u[i] = fa_u_dev(i);
          for (int i = 0; i < nlev + 1; ++i)
            mu_i[i] = mu_i_dev(i);
          for (int i = 0; i < nlev + 1; ++i)
            md_i[i] = md_i_dev(i);
          for (int i = 0; i < nlev; ++i)
            dudp[i] = dudp_dev(i);
          for (int i = 0; i < nlev; ++i)
            dddp[i] = dddp_dev(i);
          for (int i = 0; i < nlev; ++i)
            eudp[i] = eudp_dev(i);
          for (int i = 0; i < nlev; ++i)
            eddp[i] = eddp_dev(i);

          mam4::convproc::initialize_dcondt(team,
              doconvproc_extd, iflux_method, ktop, kbot, nlev, dpdry_i, fa_u,
              mu_i, md_i, chat_dev, gath_dev, conu_dev, cond_dev,
              dconudt_activa_dev, dconudt_wetdep_dev, dudp, dddp, eudp, eddp,
              dcondt_dev);

          const int iflux_method_2 = 2;
          // flip a bit to trip a check in initialize_dcondt
          mu_i[62] *= -1;
          md_i[62] *= -1;
          mam4::convproc::initialize_dcondt(team,
              doconvproc_extd, iflux_method_2, ktop, kbot, nlev, dpdry_i, fa_u,
              mu_i, md_i, chat_dev, gath_dev, conu_dev, cond_dev,
              dconudt_activa_dev, dconudt_wetdep_dev, dudp, dddp, eudp, eddp,
              dcondt_dev_2);
        });
    // Check case of iflux_method == 2 which is not part of the e3sm tests.
    set_host("dcondt", nlev, mam4::ConvProc::pcnst_extd, dcondt_host_2,
             dcondt_dev_2);
    const std::map<int, Real> check_vals = {
        {3782, -7310.292049285079}, {3787, -16982.38670267153},
        {3799, 31.81635154330214},  {3822, 6423.064850051624},
        {3827, 1141.976265447629},  {3835, 1.562336372012505},
        {3862, -4556.738178501109}, {3867, -539.039361569559},
        {3875, -1.268179748794017}, {3879, -1.373535224269113},
        {3942, -864.0456520073034}, {3947, 10285.26167932795},
        {3959, 44.18910783699215},  {4022, -230.2033875040762},
        {4027, 16138.44355461998},  {4039, 18.84244513302622},
        {4102, -127.1649429350546}, {4107, 2188.041013681148},
        {4119, 41.62726183580788},  {4182, -90.36295467510922},
        {4187, -2293.757346603896}, {4199, 24.92088413945955},
        {4262, -66.46657717693712}, {4267, -4122.195795472179},
        {4279, 4.82020187814818},   {4342, -85.58108483240539},
        {4347, -3235.512411591856}, {4422, -74.03610221286269},
        {4427, -941.0707192968931}, {4439, 2.710206649231272},
        {4502, -144.172781841651},  {4507, 2706.564052087646},
        {4519, -1.007888663630276}, {4582, -305.2735822443682},
        {4587, 9545.901312278957},  {4599, 6.007912488387871},
        {4662, -357.6099449674683}, {4667, 6083.516009044935},
        {4679, 3.876521828094159},  {4742, -368.9550818594454},
        {4747, 3214.831427699431},  {4822, -386.711844890242},
        {4827, 1770.294937079342},  {4839, -5.577899382867209},
        {4902, 54572.54202194751},  {4907, -69104.79437923993},
        {4915, 17.35397572012981},  {4919, -922.8108747624234},
        {4942, -44237.8022752386},  {4947, -9121.896357279784},
        {4955, -15.13095401033148}, {4982, -56940.08808061839},
        {4987, 75649.94625557365},  {4995, -18.67776122682817},
        {4999, 940.4290487384062},  {5022, 45720.73680200587},
        {5027, 9427.679518861254},  {5035, 15.63817211222605},
        {5062, -142.2996676009982}, {5067, -3707.928785583305},
        {5079, -8.547142908624165}, {5142, -84.6151145307276},
        {5147, -6632.941784558536}, {5159, -16.91386537169804},
        {5222, -168.2387026586199}, {5227, -16604.2102099751},
        {5239, -21.16186459435376}, {5302, -312.7389251096075},
        {5307, -47413.43420240164}, {5319, -1.334976456988455},
        {5382, -154.1093853174021}, {5387, -17911.74930961148},
        {5399, -8.629116662072764}, {5462, 1756.048426296213},
        {5467, 24449.33130517575},  {5475, 1.133303392684191},
        {5479, -174.7342983881855}, {5542, 2400.848508212621},
        {5547, 20678.20312956569},  {5555, 1.175814149817009},
        {5559, -212.0981876316918}, {5622, 886.3495482749365},
        {5627, 4644.804458599396},  {5639, -243.8633988644449}};
    Real norm = 0;
    for (const auto x : dcondt_host_2)
      norm += x * x;
    if (norm)
      for (const auto x : check_vals)
        EKAT_REQUIRE(std::abs(x.second - dcondt_host_2[x.first]) < .0000001);

    set_output(output, "dcondt", nlev, mam4::ConvProc::pcnst_extd, dcondt_host,
               dcondt_dev);
  });
}
