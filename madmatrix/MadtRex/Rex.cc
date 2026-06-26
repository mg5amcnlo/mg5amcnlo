/***
 *    ______
 *    | ___ \
 *    | |_/ /_____  __
 *    |    // _ \ \/ /
 *    | |\ \  __/>  <
 *    \_| \_\___/_/\_\
 *
 ***/
//
// *R*apid *e*vent e*x*traction Version 1.0.0
// Rex is a C++ library for parsing and manipulating Les Houches Event-format (LHE) files.
// It is designed to fast and lightweight, in comparison to internal parsers in programs like MadGraph.
// Currently, Rex is in development and may not contain all features necessary for full LHE parsing.
//
// Copyright © 2023-2026 CERN, CERN Author Zenny Wettersten.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// All rights not expressly granted are reserved.
//

#ifndef _REX_CC_
#define _REX_CC_

#include "Rex.h"

namespace REX
{

    std::string to_upper(const std::string &str)
    {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::toupper);
        return result;
    }

    // Generic warning function for printing warnings without throwing anything
    void warning(std::string message)
    {
        std::cout << "\n\033[1;33mWarning: ";
        std::cout << message;
        std::cout << "\033[0m\n";
    }

    // Explicit instantiation of templated functions and structs/classes
    template std::shared_ptr<std::vector<size_t>> ind_sort<int>(const std::vector<int> &vector, std::function<bool(const int &, const int &)> comp);
    template std::shared_ptr<std::vector<size_t>> ind_sort<double>(const std::vector<double> &vector, std::function<bool(const double &, const double &)> comp);

    template int ctoi<std::string>(std::string str);
    template int ctoi<std::string_view>(std::string_view str);

    template double ctod<std::string>(std::string str);
    template double ctod<std::string_view>(std::string_view str);

    template std::shared_ptr<std::vector<double>> vec_elem_mult(const std::vector<double> &vec1, const std::vector<double> &vec2);
    template std::shared_ptr<std::vector<float>> vec_elem_mult(const std::vector<float> &vec1, const std::vector<float> &vec2);
    template std::shared_ptr<std::vector<int>> vec_elem_mult(const std::vector<int> &vec1, const std::vector<int> &vec2);

    template std::vector<int> subvector<int>(std::vector<int> original, size_t begin, size_t end);
    template std::vector<size_t> subvector<size_t>(std::vector<size_t> original, size_t begin, size_t end);
    template std::vector<short int> subvector<short int>(std::vector<short int> original, size_t begin, size_t end);
    template std::vector<long int> subvector<long int>(std::vector<long int> original, size_t begin, size_t end);
    template std::vector<double> subvector<double>(std::vector<double> original, size_t begin, size_t end);
    template std::vector<float> subvector<float>(std::vector<float> original, size_t begin, size_t end);
    template std::vector<std::string> subvector<std::string>(std::vector<std::string> original, size_t begin, size_t end);
    template std::vector<std::string_view> subvector<std::string_view>(std::vector<std::string_view> original, size_t begin, size_t end);

    template struct arrN<short int, 2>;
    template struct arrN<long int, 2>;
    template struct arrN<int, 2>;
    template struct arrN<float, 2>;
    template struct arrN<double, 2>;
    template struct arrN<short int, 3>;
    template struct arrN<long int, 3>;
    template struct arrN<int, 3>;
    template struct arrN<float, 3>;
    template struct arrN<double, 3>;
    template struct arrN<short int, 4>;
    template struct arrN<long int, 4>;
    template struct arrN<int, 4>;
    template struct arrN<float, 4>;
    template struct arrN<double, 4>;

    template struct arrNRef<short int, 2>;
    template struct arrNRef<long int, 2>;
    template struct arrNRef<int, 2>;
    template struct arrNRef<float, 2>;
    template struct arrNRef<double, 2>;
    template struct arrNRef<short int, 3>;
    template struct arrNRef<long int, 3>;
    template struct arrNRef<int, 3>;
    template struct arrNRef<float, 3>;
    template struct arrNRef<double, 3>;
    template struct arrNRef<short int, 4>;
    template struct arrNRef<long int, 4>;
    template struct arrNRef<int, 4>;
    template struct arrNRef<float, 4>;
    template struct arrNRef<double, 4>;

    template struct vecArrN<short int, 2>;
    template struct vecArrN<long int, 2>;
    template struct vecArrN<int, 2>;
    template struct vecArrN<float, 2>;
    template struct vecArrN<double, 2>;
    template struct vecArrN<short int, 3>;
    template struct vecArrN<long int, 3>;
    template struct vecArrN<int, 3>;
    template struct vecArrN<float, 3>;
    template struct vecArrN<double, 3>;
    template struct vecArrN<short int, 4>;
    template struct vecArrN<long int, 4>;
    template struct vecArrN<int, 4>;
    template struct vecArrN<float, 4>;
    template struct vecArrN<double, 4>;

    // Parton getters
    arr4<double> &parton::momenta() { return this->momenta_; }
    const arr4<double> &parton::momenta() const { return this->momenta_; }
    arr4<double> &parton::momentum() { return this->momenta_; }
    const arr4<double> &parton::momentum() const { return this->momenta_; }
    arr4<double> &parton::pUP() { return this->momenta_; }
    const arr4<double> &parton::pUP() const { return this->momenta_; }
    arr4<double> &parton::p() { return this->momenta_; }
    const arr4<double> &parton::p() const { return this->momenta_; }
    arr4<double> &parton::mom() { return this->momenta_; }
    const arr4<double> &parton::mom() const { return this->momenta_; }
    double &parton::E() { return this->momenta_[0]; }
    const double &parton::E() const { return this->momenta_[0]; }
    double &parton::t() { return this->momenta_[0]; }
    const double &parton::t() const { return this->momenta_[0]; }
    double &parton::px() { return this->momenta_[1]; }
    const double &parton::px() const { return this->momenta_[1]; }
    double &parton::x() { return this->momenta_[1]; }
    const double &parton::x() const { return this->momenta_[1]; }
    double &parton::py() { return this->momenta_[2]; }
    const double &parton::py() const { return this->momenta_[2]; }
    double &parton::y() { return this->momenta_[2]; }
    const double &parton::y() const { return this->momenta_[2]; }
    double &parton::pz() { return this->momenta_[3]; }
    const double &parton::pz() const { return this->momenta_[3]; }
    double &parton::z() { return this->momenta_[3]; }
    const double &parton::z() const { return this->momenta_[3]; }
    double &parton::m() { return this->mass_; }
    const double &parton::m() const { return this->mass_; }
    double &parton::mass() { return this->mass_; }
    const double &parton::mass() const { return this->mass_; }
    double &parton::vtim() { return this->vtim_; }
    const double &parton::vtim() const { return this->vtim_; }
    double &parton::vTimUP() { return this->vtim_; }
    const double &parton::vTimUP() const { return this->vtim_; }
    double &parton::spin() { return this->spin_; }
    const double &parton::spin() const { return this->spin_; }
    double &parton::spinUP() { return this->spin_; }
    const double &parton::spinUP() const { return this->spin_; }
    long int &parton::pdg() { return this->pdg_; }
    const long int &parton::pdg() const { return this->pdg_; }
    long int &parton::idUP() { return this->pdg_; }
    const long int &parton::idUP() const { return this->pdg_; }
    long int &parton::id() { return this->pdg_; }
    const long int &parton::id() const { return this->pdg_; }
    short int &parton::status() { return this->status_; }
    const short int &parton::status() const { return this->status_; }
    short int &parton::iStUP() { return this->status_; }
    const short int &parton::iStUP() const { return this->status_; }
    short int &parton::iSt() { return this->status_; }
    const short int &parton::iSt() const { return this->status_; }
    arr2<short int> &parton::mother() { return this->mother_; }
    const arr2<short int> &parton::mother() const { return this->mother_; }
    arr2<short int> &parton::mothUP() { return this->mother_; }
    const arr2<short int> &parton::mothUP() const { return this->mother_; }
    arr2<short int> &parton::moth() { return this->mother_; }
    const arr2<short int> &parton::moth() const { return this->mother_; }
    arr2<short int> &parton::icol() { return this->icol_; }
    const arr2<short int> &parton::icol() const { return this->icol_; }
    arr2<short int> &parton::iColUP() { return this->icol_; }
    const arr2<short int> &parton::iColUP() const { return this->icol_; }
    arr2<short int> &parton::iCol() { return this->icol_; }
    const arr2<short int> &parton::iCol() const { return this->icol_; }

    parton &parton::set_momenta(const arr4<double> &mom)
    {
        this->momenta_ = mom;
        return *this;
    }
    parton &parton::set_mom(const arr4<double> &mom) { return this->set_momenta(mom); }
    parton &parton::set_pUP(const arr4<double> &mom) { return this->set_momenta(mom); }
    parton &parton::set_p(const arr4<double> &mom) { return this->set_momenta(mom); }

    parton &parton::set_E(double E)
    {
        this->momenta_[0] = E;
        return *this;
    }
    parton &parton::set_t(double pt) { return this->set_E(pt); }

    parton &parton::set_x(double x)
    {
        this->momenta_[1] = x;
        return *this;
    }
    parton &parton::set_px(double px) { return this->set_x(px); }

    parton &parton::set_y(double y)
    {
        this->momenta_[2] = y;
        return *this;
    }
    parton &parton::set_py(double py) { return this->set_y(py); }

    parton &parton::set_z(double z)
    {
        this->momenta_[3] = z;
        return *this;
    }
    parton &parton::set_pz(double pz) { return this->set_z(pz); }

    parton &parton::set_mass(double m)
    {
        this->mass_ = m;
        return *this;
    }

    parton &parton::set_vtim(double v)
    {
        this->vtim_ = v;
        return *this;
    }

    parton &parton::set_vTimUP(double v)
    {
        return this->set_vtim(v);
    }

    parton &parton::set_spin(double s)
    {
        this->spin_ = s;
        return *this;
    }

    parton &parton::set_spinUP(double s)
    {
        return this->set_spin(s);
    }

    parton &parton::set_pdg(long int p)
    {
        this->pdg_ = p;
        return *this;
    }

    parton &parton::set_id(long int id)
    {
        return this->set_pdg(id);
    }

    parton &parton::set_idUP(long int id)
    {
        return this->set_pdg(id);
    }

    parton &parton::set_status(short int st)
    {
        this->status_ = st;
        return *this;
    }

    parton &parton::set_iStUP(short int st)
    {
        return this->set_status(st);
    }

    parton &parton::set_iSt(short int st)
    {
        return this->set_status(st);
    }

    parton &parton::set_mother(const arr2<short int> &mth)
    {
        this->mother_ = mth;
        return *this;
    }

    parton &parton::set_mothUP(const arr2<short int> &mth)
    {
        return this->set_mother(mth);
    }

    parton &parton::set_moth(const arr2<short int> &mth)
    {
        return this->set_mother(mth);
    }

    parton &parton::set_mother(const short int m1, const short int m2)
    {
        this->mother_[0] = m1;
        this->mother_[1] = m2;
        return *this;
    }

    parton &parton::set_mothUP(const short int m1, const short int m2)
    {
        return this->set_mother(m1, m2);
    }

    parton &parton::set_moth(const short int m1, const short int m2)
    {
        return this->set_mother(m1, m2);
    }

    parton &parton::set_icol(const arr2<short int> &icol)
    {
        this->icol_ = icol;
        return *this;
    }

    parton &parton::set_iColUP(const arr2<short int> &icol)
    {
        return this->set_icol(icol);
    }

    parton &parton::set_iCol(const arr2<short int> &col)
    {
        return this->set_icol(col);
    }

    parton &parton::set_icol(const short int i1, const short int i2)
    {
        this->icol_[0] = i1;
        this->icol_[1] = i2;
        return *this;
    }

    parton &parton::set_iColUP(const short int i1, const short int i2)
    {
        return this->set_icol(i1, i2);
    }

    parton &parton::set_iCol(const short int i1, const short int i2)
    {
        return this->set_icol(i1, i2);
    }

    // Physical observables
    double parton::pT() const
    {
        return std::sqrt(this->momenta_[1] * this->momenta_[1] + this->momenta_[2] * this->momenta_[2]);
    }
    double parton::pT2() const
    {
        return this->momenta_[1] * this->momenta_[1] + this->momenta_[2] * this->momenta_[2];
    }
    double parton::pL() const
    {
        return this->momenta_[3];
    }
    double parton::pL2() const
    {
        return this->momenta_[3] * this->momenta_[3];
    }
    double parton::eT() const
    {
        return std::sqrt(this->mass_ * this->mass_ + this->pT2());
    }
    double parton::eT2() const
    {
        return this->mass_ * this->mass_ + this->pT2();
    }
    double parton::phi() const
    {
        return std::atan2(this->momenta_[2], this->momenta_[1]);
    }
    double parton::theta() const
    {
        double p = std::sqrt(this->momenta_[1] * this->momenta_[1] + this->momenta_[2] * this->momenta_[2] + this->momenta_[3] * this->momenta_[3]);
        if (p == 0.0)
            return 0.0;
        return std::acos(this->momenta_[3] / p);
    }
    double parton::eta() const
    {
        double p = std::sqrt(this->momenta_[1] * this->momenta_[1] + this->momenta_[2] * this->momenta_[2] + this->momenta_[3] * this->momenta_[3]);
        if (std::abs(p - std::abs(this->momenta_[3])) < 1e-10)
            return (this->momenta_[3] >= 0.0) ? INFINITY : -INFINITY; // Infinite pseudorapidity for massless particles along the beamline

        return 0.5 * std::log((p + this->momenta_[3]) / (p - this->momenta_[3]));
    }
    double parton::rap() const
    {
        if (std::abs(this->momenta_[0] - std::abs(this->momenta_[3])) < 1e-10)
            return (this->momenta_[3] >= 0.0) ? INFINITY : -INFINITY; // Infinite rapidity for massless particles along the beamline

        return 0.5 * std::log((this->momenta_[0] + this->momenta_[3]) / (this->momenta_[0] - this->momenta_[3]));
    }
    double parton::mT() const
    {
        return std::sqrt(this->mass_ * this->mass_ + this->pT2());
    }
    double parton::mT2() const
    {
        return this->mass_ * this->mass_ + this->pT2();
    }
    double parton::m2() const
    {
        return this->mass_ * this->mass_;
    }

    // Print functions
    void event::particle::print(std::ostream &os) const
    {
        os << std::setprecision(10) << std::scientific << std::noshowpos // Use scientific notation with 11 digits printed, don't include leading +
           << " " << std::setw(8) << this->pdg_
           << " " << std::setw(2) << this->status_
           << " " << std::setw(4) << this->mother_[0]
           << " " << std::setw(4) << this->mother_[1]
           << " " << std::setw(4) << this->icol_[0]
           << " " << std::setw(4) << this->icol_[1]
           << std::showpos                               // Enable leading + for spatial momenta
           << " " << std::setw(16) << this->momentum_[1] // Note: momentum ordering in LHEF is (px,py,pz,E) whereas we store it as (E,px,py,pz)
           << " " << std::setw(16) << this->momentum_[2]
           << " " << std::setw(16) << this->momentum_[3]
           << std::noshowpos // Disable leading + for energy, mass, lifetime, spin (should all be positive)
           << " " << std::setw(16) << this->momentum_[0]
           << " " << std::setw(16) << this->mass_
           << std::setprecision(4) // Lower precision for lifetime, spin
           << " " << std::setw(10) << this->vtim_
           << " " << std::setw(10) << this->spin_ << "\n";
    }

    void event::const_particle::print(std::ostream &os) const
    {
        os << std::setprecision(10) << std::scientific << std::noshowpos // Use scientific notation with 11 digits printed, don't include leading +
           << " " << std::setw(8) << this->pdg_
           << " " << std::setw(2) << this->status_
           << " " << std::setw(4) << this->mother_[0]
           << " " << std::setw(4) << this->mother_[1]
           << " " << std::setw(4) << this->icol_[0]
           << " " << std::setw(4) << this->icol_[1]
           << std::showpos                               // Enable leading + for spatial momenta
           << " " << std::setw(16) << this->momentum_[1] // Note: momentum ordering in LHEF is (px,py,pz,E) whereas we store it as (E,px,py,pz)
           << " " << std::setw(16) << this->momentum_[2]
           << " " << std::setw(16) << this->momentum_[3]
           << std::noshowpos // Disable leading + for energy, mass, lifetime, spin (should all be positive)
           << " " << std::setw(16) << this->momentum_[0]
           << " " << std::setw(16) << this->mass_
           << std::setprecision(4) // Lower precision for lifetime, spin
           << " " << std::setw(10) << this->vtim_
           << " " << std::setw(10) << this->spin_ << "\n";
    }

    event::event(size_t n_particles)
        : momenta_(n_particles),
          mass_(n_particles),
          vtim_(n_particles),
          spin_(n_particles),
          pdg_(n_particles),
          status_(n_particles),
          mother_(n_particles),
          icol_(n_particles)
    {
        this->n_ = n_particles;
    }

    event::event(std::vector<parton> particles)
        : momenta_(particles.size()),
          mass_(particles.size()),
          vtim_(particles.size()),
          spin_(particles.size()),
          pdg_(particles.size()),
          status_(particles.size()),
          mother_(particles.size()),
          icol_(particles.size())
    {
        this->n_ = particles.size();
        for (size_t i = 0; i < this->n_; ++i)
        {
            momenta_[i] = particles[i].momenta_;
            mass_[i] = particles[i].mass_;
            vtim_[i] = particles[i].vtim_;
            spin_[i] = particles[i].spin_;
            pdg_[i] = particles[i].pdg_;
            status_[i] = particles[i].status_;
            mother_[i] = particles[i].mother_;
            icol_[i] = particles[i].icol_;
        }
    }

    void event::print_head(std::ostream &os) const
    {
        os << std::setprecision(7) << std::scientific << std::noshowpos
           << std::left // Left-align n
           << " " << std::setw(4) << this->n_
           << std::right // Right-align remaining info
           << " " << std::setw(3) << this->proc_id_
           << std::showpos // Enable leading + for weight
           << " " << std::setw(13) << this->weight_
           << std::noshowpos << std::setprecision(8) // Disable leading + for scale and couplings
           << " " << std::setw(14) << this->scale_
           << " " << std::setw(14) << this->alphaEW_
           << " " << std::setw(14) << this->alphaS_ << "\n";
    }

    void event::print_wgts_no_ids(std::ostream &os) const
    {
        if (this->wgts_.size() == 0)
            return;
        os << std::setprecision(7) << std::scientific << std::showpos;
        os << "<weights> ";
        for (const auto &w : this->wgts_)
        {
            os << w << " ";
        }
        os << "</weights>\n";
    }

    void event::print_wgts_ids(std::ostream &os) const
    {
        if (this->weight_ids->size() > this->wgts_.size())
        {
            warning("More weight IDs than weights available. Printing zero weights, which may have incorrect indexing.");
            // this->wgts_.resize(this->weight_ids->size(), 0.0);
        }
        if (this->weight_ids->size() < this->wgts_.size())
        {
            for (size_t i = this->weight_ids->size(); i < this->wgts_.size(); ++i)
            {
                this->weight_ids->push_back("rwgt_" + std::to_string(i + 1));
            }
        }
        if (this->wgts_.size() == 0)
            return;
        os << std::setprecision(7) << std::scientific << std::showpos;
        os << "<rwgt>";
        for (size_t i = 0; i < this->weight_ids->size(); ++i)
        {
            os << "\n<wgt id=\'" << (*this->weight_ids)[i] << "\'> ";
            if (i < this->wgts_.size())
            {
                os << this->wgts_[i];
            }
            else
            {
                os << 0.0;
            }
            os << " </wgt>";
        }
        os << "\n</rwgt>\n";
    }

    void event::print_wgts(std::ostream &os, bool include_ids) const
    {
        if (include_ids)
        {
            print_wgts_ids(os);
        }
        else
        {
            print_wgts_no_ids(os);
        }
    }

    double event::gS()
    {
        return std::sqrt(4. * pi * alphaS_);
    }

    double event::get_muF() const
    {
        return muF_ != 0.0 ? muF_ : scale_;
    }

    double event::get_muR() const
    {
        return muR_ != 0.0 ? muR_ : scale_;
    }

    double event::get_muPS() const
    {
        return muPS_ != 0.0 ? muPS_ : scale_;
    }

    size_t &event::nUP() { return n_; }
    const size_t &event::nUP() const { return n_; }
    size_t &event::n() { return n_; }
    const size_t &event::n() const { return n_; }

    long int &event::idPrUP() { return proc_id_; }
    const long int &event::idPrUP() const { return proc_id_; }
    long int &event::idPr() { return proc_id_; }
    const long int &event::idPr() const { return proc_id_; }

    double &event::xWgtUP() { return weight_; }
    const double &event::xWgtUP() const { return weight_; }
    double &event::xWgt() { return weight_; }
    const double &event::xWgt() const { return weight_; }
    double &event::weight() { return weight_; }
    const double &event::weight() const { return weight_; }

    double &event::scalUP() { return scale_; }
    const double &event::scalUP() const { return scale_; }
    double &event::scale() { return scale_; }
    const double &event::scale() const { return scale_; }
    double &event::muF() { return muF_; }
    const double &event::muF() const { return muF_; }
    double &event::muR() { return muR_; }
    const double &event::muR() const { return muR_; }
    double &event::muPS() { return muPS_; }
    const double &event::muPS() const { return muPS_; }

    double &event::aQEDUP() { return alphaEW_; }
    const double &event::aQEDUP() const { return alphaEW_; }
    double &event::aQED() { return alphaEW_; }
    const double &event::aQED() const { return alphaEW_; }
    double &event::alphaEW() { return alphaEW_; }
    const double &event::alphaEW() const { return alphaEW_; }
    double &event::aEW() { return alphaEW_; }
    const double &event::aEW() const { return alphaEW_; }

    double &event::aQCDUP() { return alphaS_; }
    const double &event::aQCDUP() const { return alphaS_; }
    double &event::aQCD() { return alphaS_; }
    const double &event::aQCD() const { return alphaS_; }
    double &event::alphaS() { return alphaS_; }
    const double &event::alphaS() const { return alphaS_; }
    double &event::aS() { return alphaS_; }
    const double &event::aS() const { return alphaS_; }

    vecArr4<double> &event::pUP() { return momenta_; }
    const vecArr4<double> &event::pUP() const { return momenta_; }
    vecArr4<double> &event::p() { return momenta_; }
    const vecArr4<double> &event::p() const { return momenta_; }
    vecArr4<double> &event::momenta() { return momenta_; }
    const vecArr4<double> &event::momenta() const { return momenta_; }
    vecArr4<double> &event::momentum() { return momenta_; }
    const vecArr4<double> &event::momentum() const { return momenta_; }

    std::vector<double> &event::mUP() { return mass_; }
    const std::vector<double> &event::mUP() const { return mass_; }
    std::vector<double> &event::m() { return mass_; }
    const std::vector<double> &event::m() const { return mass_; }
    std::vector<double> &event::mass() { return mass_; }
    const std::vector<double> &event::mass() const { return mass_; }

    std::vector<double> &event::vTimUP() { return vtim_; }
    const std::vector<double> &event::vTimUP() const { return vtim_; }
    std::vector<double> &event::vtim() { return vtim_; }
    const std::vector<double> &event::vtim() const { return vtim_; }
    std::vector<double> &event::vTim() { return vtim_; }
    const std::vector<double> &event::vTim() const { return vtim_; }

    std::vector<double> &event::spinUP() { return spin_; }
    const std::vector<double> &event::spinUP() const { return spin_; }
    std::vector<double> &event::spin() { return spin_; }
    const std::vector<double> &event::spin() const { return spin_; }

    std::vector<long int> &event::idUP() { return pdg_; }
    const std::vector<long int> &event::idUP() const { return pdg_; }
    std::vector<long int> &event::id() { return pdg_; }
    const std::vector<long int> &event::id() const { return pdg_; }
    std::vector<long int> &event::pdg() { return pdg_; }
    const std::vector<long int> &event::pdg() const { return pdg_; }

    std::vector<short int> &event::iStUP() { return status_; }
    const std::vector<short int> &event::iStUP() const { return status_; }
    std::vector<short int> &event::iSt() { return status_; }
    const std::vector<short int> &event::iSt() const { return status_; }
    std::vector<short int> &event::status() { return status_; }
    const std::vector<short int> &event::status() const { return status_; }

    vecArr2<short int> &event::mothUP() { return mother_; }
    const vecArr2<short int> &event::mothUP() const { return mother_; }
    vecArr2<short int> &event::moth() { return mother_; }
    const vecArr2<short int> &event::moth() const { return mother_; }
    vecArr2<short int> &event::mother() { return mother_; }
    const vecArr2<short int> &event::mother() const { return mother_; }

    vecArr2<short int> &event::iColUP() { return icol_; }
    const vecArr2<short int> &event::iColUP() const { return icol_; }
    vecArr2<short int> &event::iCol() { return icol_; }
    const vecArr2<short int> &event::iCol() const { return icol_; }
    vecArr2<short int> &event::icol() { return icol_; }
    const vecArr2<short int> &event::icol() const { return icol_; }

    std::vector<double> &event::wgts() { return wgts_; }
    const std::vector<double> &event::wgts() const { return wgts_; }

    size_t event::n_wgts() const { return wgts_.size(); }

    event &event::set_n(size_t n_particles)
    {
        if (n_particles != this->n_)
        {
            // Resize all vectors to match the new number of particles
            // (this may overwrite existing particles)
            momenta_.resize(n_particles);
            mass_.resize(n_particles);
            vtim_.resize(n_particles);
            spin_.resize(n_particles);
            pdg_.resize(n_particles);
            status_.resize(n_particles);
            mother_.resize(n_particles);
            icol_.resize(n_particles);
        }
        this->n_ = n_particles;
        return *this;
    }

    event &event::set_proc_id(long int proc_id)
    {
        this->proc_id_ = proc_id;
        return *this;
    }

    event &event::set_weight(double weight)
    {
        this->weight_ = weight;
        return *this;
    }

    event &event::set_scale(double scale)
    {
        this->scale_ = scale;
        this->muF_ = (muF_ == 0.0) ? scale : muF_;    // Set muF if not already set
        this->muR_ = (muR_ == 0.0) ? scale : muR_;    // Set muR if not already set
        this->muPS_ = (muPS_ == 0.0) ? scale : muPS_; // Set muPS if not already set
        return *this;
    }

    event &event::set_muF(double muF)
    {
        this->muF_ = muF;
        return *this;
    }

    event &event::set_muR(double muR)
    {
        this->muR_ = muR;
        return *this;
    }

    event &event::set_muPS(double muPS)
    {
        this->muPS_ = muPS;
        return *this;
    }

    event &event::set_alphaEW(double alphaEW)
    {
        this->alphaEW_ = alphaEW;
        return *this;
    }

    event &event::set_alphaS(double alphaS)
    {
        this->alphaS_ = alphaS;
        return *this;
    }

    event &event::set_momenta(const vecArr4<double> &mom)
    {
        this->momenta_ = mom;
        return *this;
    }

    event &event::set_momenta(const std::vector<std::array<double, 4>> &mom)
    {
        this->momenta_.resize(mom.size());
        for (size_t i = 0; i < mom.size(); ++i)
        {
            this->momenta_[i] = {mom[i][0], mom[i][1], mom[i][2], mom[i][3]};
        }
        return *this;
    }

    event &event::set_mass(const std::vector<double> &m)
    {
        this->mass_ = m;
        return *this;
    }

    event &event::set_vtim(const std::vector<double> &v)
    {
        this->vtim_ = v;
        return *this;
    }

    event &event::set_spin(const std::vector<double> &s)
    {
        this->spin_ = s;
        return *this;
    }

    event &event::set_pdg(const std::vector<long int> &p)
    {
        this->pdg_ = p;
        return *this;
    }

    event &event::set_status(const std::vector<short int> &st)
    {
        this->status_ = st;
        return *this;
    }

    event &event::set_mother(const vecArr2<short int> &m)
    {
        this->mother_ = m;
        return *this;
    }

    event &event::set_mother(const std::vector<std::array<short int, 2>> &m)
    {
        this->mother_.resize(m.size());
        for (size_t i = 0; i < m.size(); ++i)
        {
            this->mother_[i] = {m[i][0], m[i][1]};
        }
        return *this;
    }

    event &event::set_icol(const vecArr2<short int> &c)
    {
        this->icol_ = c;
        return *this;
    }

    event &event::set_icol(const std::vector<std::array<short int, 2>> &c)
    {
        this->icol_.resize(c.size());
        for (size_t i = 0; i < c.size(); ++i)
        {
            this->icol_[i] = {c[i][0], c[i][1]};
        }
        return *this;
    }

    event &event::set_wgts(const std::vector<double> &w)
    {
        this->wgts_ = w;
        return *this;
    }

    event &event::add_wgt(double w, const std::string &id)
    {
        this->wgts_.push_back(w);
        if (!id.empty())
            this->weight_ids->push_back(id);
        return *this;
    }

    event &event::add_particle(const parton &p)
    {
        momenta_.push_back(p.momenta_);
        mass_.push_back(p.mass_);
        vtim_.push_back(p.vtim_);
        spin_.push_back(p.spin_);
        pdg_.push_back(p.pdg_);
        status_.push_back(p.status_);
        mother_.push_back(p.mother_);
        icol_.push_back(p.icol_);
        ++n_;
        return *this;
    }

    event &event::add_particle(const particle &p)
    {
        momenta_.push_back(p.momentum_);
        mass_.push_back(p.mass_);
        vtim_.push_back(p.vtim_);
        spin_.push_back(p.spin_);
        pdg_.push_back(p.pdg_);
        status_.push_back(p.status_);
        mother_.push_back(p.mother_);
        icol_.push_back(p.icol_);
        ++n_;
        return *this;
    }

    event &event::add_particle(const const_particle &p)
    {
        momenta_.push_back(p.momentum_);
        mass_.push_back(p.mass_);
        vtim_.push_back(p.vtim_);
        spin_.push_back(p.spin_);
        pdg_.push_back(p.pdg_);
        status_.push_back(p.status_);
        mother_.push_back(p.mother_);
        icol_.push_back(p.icol_);
        ++n_;
        return *this;
    }

    // Accessors
    event::particle event::get_particle(size_t i)
    {
        return {momenta_.at(i), mass_.at(i), vtim_.at(i), spin_.at(i),
                pdg_.at(i), status_.at(i), mother_.at(i), icol_.at(i)};
    }

    event::const_particle event::get_particle(size_t i) const
    {
        return {momenta_.at(i), mass_.at(i), vtim_.at(i), spin_.at(i),
                pdg_.at(i), status_.at(i), mother_.at(i), icol_.at(i)};
    }

    size_t event::size() const
    {
        validate();
        return momenta_.size();
    }

    // Iterators
    event::particle event::particle_iterator::operator*()
    {
        return evt->get_particle(index);
    }
    event::particle_iterator &event::particle_iterator::operator++()
    {
        ++index;
        return *this;
    }
    bool event::particle_iterator::operator!=(const particle_iterator &other) const
    {
        return index != other.index;
    }

    event::const_particle event::const_particle_iterator::operator*() const
    {
        return evt->get_particle(index);
    }
    event::const_particle_iterator &event::const_particle_iterator::operator++()
    {
        ++index;
        return *this;
    }
    bool event::const_particle_iterator::operator!=(const const_particle_iterator &other) const
    {
        return index != other.index;
    }

    event::particle_iterator event::begin() { return {this, 0}; }
    event::particle_iterator event::end() { return {this, size()}; }

    event::const_particle_iterator event::begin() const { return {this, 0}; }
    event::const_particle_iterator event::end() const { return {this, size()}; }

    // Particle level accessors and setters
    arr4Ref<double> event::particle::pUP() { return momentum_; }
    arr4Ref<const double> event::particle::pUP() const { return arr4Ref<const double>{momentum_.p}; }
    arr4Ref<double> event::particle::p() { return momentum_; }
    arr4Ref<const double> event::particle::p() const { return arr4Ref<const double>{momentum_.p}; }
    arr4Ref<double> event::particle::mom() { return momentum_; }
    arr4Ref<const double> event::particle::mom() const { return arr4Ref<const double>{momentum_.p}; }
    arr4Ref<double> event::particle::momentum() { return momentum_; }
    arr4Ref<const double> event::particle::momentum() const { return arr4Ref<const double>{momentum_.p}; }
    double &event::particle::E() { return momentum_[0]; }
    const double &event::particle::E() const { return momentum_[0]; }
    double &event::particle::t() { return momentum_[0]; }
    const double &event::particle::t() const { return momentum_[0]; }
    double &event::particle::x() { return momentum_[1]; }
    double &event::particle::px() { return momentum_[1]; }
    const double &event::particle::x() const { return momentum_[1]; }
    const double &event::particle::px() const { return momentum_[1]; }
    double &event::particle::y() { return momentum_[2]; }
    double &event::particle::py() { return momentum_[2]; }
    const double &event::particle::y() const { return momentum_[2]; }
    const double &event::particle::py() const { return momentum_[2]; }
    double &event::particle::z() { return momentum_[3]; }
    double &event::particle::pz() { return momentum_[3]; }
    const double &event::particle::z() const { return momentum_[3]; }
    const double &event::particle::pz() const { return momentum_[3]; }
    double &event::particle::mUP() { return mass_; }
    const double &event::particle::mUP() const { return mass_; }
    double &event::particle::m() { return mass_; }
    const double &event::particle::m() const { return mass_; }
    double &event::particle::mass() { return mass_; }
    const double &event::particle::mass() const { return mass_; }
    double &event::particle::vTimUP() { return vtim_; }
    const double &event::particle::vTimUP() const { return vtim_; }
    double &event::particle::vtim() { return vtim_; }
    const double &event::particle::vtim() const { return vtim_; }
    double &event::particle::vTim() { return vtim_; }
    const double &event::particle::vTim() const { return vtim_; }
    double &event::particle::spinUP() { return spin_; }
    const double &event::particle::spinUP() const { return spin_; }
    double &event::particle::spin() { return spin_; }
    const double &event::particle::spin() const { return spin_; }
    long int &event::particle::idUP() { return pdg_; }
    const long int &event::particle::idUP() const { return pdg_; }
    long int &event::particle::id() { return pdg_; }
    const long int &event::particle::id() const { return pdg_; }
    long int &event::particle::pdg() { return pdg_; }
    const long int &event::particle::pdg() const { return pdg_; }
    short int &event::particle::iStUP() { return status_; }
    const short int &event::particle::iStUP() const { return status_; }
    short int &event::particle::iSt() { return status_; }
    const short int &event::particle::iSt() const { return status_; }
    short int &event::particle::status() { return status_; }
    const short int &event::particle::status() const { return status_; }
    arr2Ref<short int> event::particle::mothUP() { return mother_; }
    const arr2Ref<short int> event::particle::mothUP() const { return mother_; }
    arr2Ref<short int> event::particle::moth() { return mother_; }
    const arr2Ref<short int> event::particle::moth() const { return mother_; }
    arr2Ref<short int> event::particle::mother() { return mother_; }
    const arr2Ref<short int> event::particle::mother() const { return mother_; }
    arr2Ref<short int> event::particle::iColUP() { return icol_; }
    const arr2Ref<short int> event::particle::iColUP() const { return icol_; }
    arr2Ref<short int> event::particle::iCol() { return icol_; }
    const arr2Ref<short int> event::particle::iCol() const { return icol_; }
    arr2Ref<short int> event::particle::icol() { return icol_; }
    const arr2Ref<short int> event::particle::icol() const { return icol_; }

    // Physical observables
    double event::particle::pT() const
    {
        return std::sqrt(this->momentum_[1] * this->momentum_[1] + this->momentum_[2] * this->momentum_[2]);
    }
    double event::particle::pT2() const
    {
        return this->momentum_[1] * this->momentum_[1] + this->momentum_[2] * this->momentum_[2];
    }
    double event::particle::pL() const
    {
        return this->momentum_[3];
    }
    double event::particle::pL2() const
    {
        return this->momentum_[3] * this->momentum_[3];
    }
    double event::particle::eT() const
    {
        return std::sqrt(this->mass_ * this->mass_ + this->pT2());
    }
    double event::particle::eT2() const
    {
        return this->mass_ * this->mass_ + this->pT2();
    }
    double event::particle::phi() const
    {
        return std::atan2(this->momentum_[2], this->momentum_[1]);
    }
    double event::particle::theta() const
    {
        double p = std::sqrt(this->momentum_[1] * this->momentum_[1] + this->momentum_[2] * this->momentum_[2] + this->momentum_[3] * this->momentum_[3]);
        if (p == 0.0)
            return 0.0;
        return std::acos(this->momentum_[3] / p);
    }
    double event::particle::eta() const
    {
        double p = std::sqrt(this->momentum_[1] * this->momentum_[1] + this->momentum_[2] * this->momentum_[2] + this->momentum_[3] * this->momentum_[3]);
        if (std::abs(p - std::abs(this->momentum_[3])) < 1e-10)
            return (this->momentum_[3] >= 0.0) ? INFINITY : -INFINITY; // Infinite pseudorapidity for massless particles along the beamline

        return 0.5 * std::log((p + this->momentum_[3]) / (p - this->momentum_[3]));
    }
    double event::particle::rap() const
    {
        if (std::abs(this->momentum_[0] - std::abs(this->momentum_[3])) < 1e-10)
            return (this->momentum_[3] >= 0.0) ? INFINITY : -INFINITY; // Infinite rapidity for massless particles along the beamline

        return 0.5 * std::log((this->momentum_[0] + this->momentum_[3]) / (this->momentum_[0] - this->momentum_[3]));
    }
    double event::particle::mT() const
    {
        return std::sqrt(this->mass_ * this->mass_ + this->pT2());
    }
    double event::particle::mT2() const
    {
        return this->mass_ * this->mass_ + this->pT2();
    }
    double event::particle::m2() const
    {
        return this->mass_ * this->mass_;
    }

    arr4Ref<const double> event::const_particle::pUP() const
    {
        return momentum_;
    }
    arr4Ref<const double> event::const_particle::mom() const
    {
        return momentum_;
    }

    arr4Ref<const double> event::const_particle::momentum() const
    {
        return momentum_;
    }

    arr4Ref<const double> event::const_particle::p() const
    {
        return momentum_;
    }

    const double &event::const_particle::E() const { return momentum_[0]; }
    const double &event::const_particle::t() const { return momentum_[0]; }
    const double &event::const_particle::x() const { return momentum_[1]; }
    const double &event::const_particle::px() const { return momentum_[1]; }
    const double &event::const_particle::y() const { return momentum_[2]; }
    const double &event::const_particle::py() const { return momentum_[2]; }
    const double &event::const_particle::z() const { return momentum_[3]; }
    const double &event::const_particle::pz() const { return momentum_[3]; }
    const double &event::const_particle::mUP() const { return mass_; }
    const double &event::const_particle::m() const { return mass_; }
    const double &event::const_particle::mass() const { return mass_; }
    const double &event::const_particle::vTimUP() const { return vtim_; }
    const double &event::const_particle::vtim() const { return vtim_; }
    const double &event::const_particle::vTim() const { return vtim_; }
    const double &event::const_particle::spinUP() const { return spin_; }
    const double &event::const_particle::spin() const { return spin_; }
    const long int &event::const_particle::idUP() const { return pdg_; }
    const long int &event::const_particle::id() const { return pdg_; }
    const long int &event::const_particle::pdg() const { return pdg_; }
    const short int &event::const_particle::iSt() const { return status_; }
    const short int &event::const_particle::status() const { return status_; }
    const short int &event::const_particle::iStUP() const { return status_; }
    arr2Ref<const short int> event::const_particle::mothUP() const { return mother_; }
    arr2Ref<const short int> event::const_particle::moth() const { return mother_; }
    arr2Ref<const short int> event::const_particle::mother() const { return mother_; }
    arr2Ref<const short int> event::const_particle::iColUP() const { return icol_; }
    arr2Ref<const short int> event::const_particle::iCol() const { return icol_; }
    arr2Ref<const short int> event::const_particle::icol() const { return icol_; }

    // Physical observables
    double event::const_particle::pT() const
    {
        return std::sqrt(this->momentum_[1] * this->momentum_[1] + this->momentum_[2] * this->momentum_[2]);
    }
    double event::const_particle::pT2() const
    {
        return this->momentum_[1] * this->momentum_[1] + this->momentum_[2] * this->momentum_[2];
    }
    double event::const_particle::pL() const
    {
        return this->momentum_[3];
    }
    double event::const_particle::pL2() const
    {
        return this->momentum_[3] * this->momentum_[3];
    }
    double event::const_particle::eT() const
    {
        return std::sqrt(this->mass_ * this->mass_ + this->pT2());
    }
    double event::const_particle::eT2() const
    {
        return this->mass_ * this->mass_ + this->pT2();
    }
    double event::const_particle::phi() const
    {
        return std::atan2(this->momentum_[2], this->momentum_[1]);
    }
    double event::const_particle::theta() const
    {
        double p = std::sqrt(this->momentum_[1] * this->momentum_[1] + this->momentum_[2] * this->momentum_[2] + this->momentum_[3] * this->momentum_[3]);
        if (p == 0.0)
            return 0.0;
        return std::acos(this->momentum_[3] / p);
    }
    double event::const_particle::eta() const
    {
        double p = std::sqrt(this->momentum_[1] * this->momentum_[1] + this->momentum_[2] * this->momentum_[2] + this->momentum_[3] * this->momentum_[3]);
        if (std::abs(p - std::abs(this->momentum_[3])) < 1e-10)
            return (this->momentum_[3] >= 0.0) ? INFINITY : -INFINITY; // Infinite pseudorapidity for massless particles along the beamline

        return 0.5 * std::log((p + this->momentum_[3]) / (p - this->momentum_[3]));
    }
    double event::const_particle::rap() const
    {
        if (std::abs(this->momentum_[0] - std::abs(this->momentum_[3])) < 1e-10)
            return (this->momentum_[3] >= 0.0) ? INFINITY : -INFINITY; // Infinite rapidity for massless particles along the beamline

        return 0.5 * std::log((this->momentum_[0] + this->momentum_[3]) / (this->momentum_[0] - this->momentum_[3]));
    }
    double event::const_particle::mT() const
    {
        return std::sqrt(this->mass_ * this->mass_ + this->pT2());
    }
    double event::const_particle::mT2() const
    {
        return this->mass_ * this->mass_ + this->pT2();
    }
    double event::const_particle::m2() const
    {
        return this->mass_ * this->mass_;
    }

    event::particle &event::particle::set_pdg(long int p)
    {
        this->pdg_ = p;
        return *this;
    }
    event::particle &event::particle::set_id(long int id) { return this->set_pdg(id); }
    event::particle &event::particle::set_idUP(long int id) { return this->set_pdg(id); }

    event::particle &event::particle::set_status(short int s)
    {
        this->status_ = s;
        return *this;
    }
    event::particle &event::particle::set_iSt(short int s) { return this->set_status(s); }
    event::particle &event::particle::set_iStUP(short int s) { return this->set_status(s); }

    event::particle &event::particle::set_mother(short int i, short int j)
    {
        this->mother_ = {i, j};
        return *this;
    }
    event::particle &event::particle::set_mother(const arr2<short int> &m)
    {
        this->mother_ = m;
        return *this;
    }
    event::particle &event::particle::set_moth(short int i, short int j) { return this->set_mother(i, j); }
    event::particle &event::particle::set_moth(const arr2<short int> &m) { return this->set_mother(m); }
    event::particle &event::particle::set_mothUP(short int i, short int j) { return this->set_mother(i, j); }
    event::particle &event::particle::set_mothUP(const arr2<short int> &m) { return this->set_mother(m); }

    event::particle &event::particle::set_icol(short int i, short int c)
    {
        this->icol_ = {i, c};
        return *this;
    }
    event::particle &event::particle::set_icol(const arr2<short int> &c)
    {
        this->icol_ = c;
        return *this;
    }
    event::particle &event::particle::set_iColUP(short int i, short int c) { return this->set_icol(i, c); }
    event::particle &event::particle::set_iColUP(const arr2<short int> &c) { return this->set_icol(c); }

    event::particle &event::particle::set_momentum(double e, double px, double py, double pz)
    {
        this->momentum_ = {e, px, py, pz};
        return *this;
    }
    event::particle &event::particle::set_momentum(const arr4<double> &mom)
    {
        this->momentum_ = mom;
        return *this;
    }
    event::particle &event::particle::set_mom(double e, double px, double py, double pz) { return this->set_momentum(e, px, py, pz); }
    event::particle &event::particle::set_mom(const arr4<double> &mom) { return this->set_momentum(mom); }
    event::particle &event::particle::set_pUP(double e, double px, double py, double pz) { return this->set_momentum(e, px, py, pz); }
    event::particle &event::particle::set_pUP(const arr4<double> &mom) { return this->set_momentum(mom); }
    event::particle &event::particle::set_p(double e, double px, double py, double pz) { return this->set_momentum(e, px, py, pz); }
    event::particle &event::particle::set_p(const arr4<double> &mom) { return this->set_momentum(mom); }

    event::particle &event::particle::set_E(double e)
    {
        this->momentum_[0] = e;
        return *this;
    }
    event::particle &event::particle::set_px(double px)
    {
        this->momentum_[1] = px;
        return *this;
    }
    event::particle &event::particle::set_py(double py)
    {
        this->momentum_[2] = py;
        return *this;
    }
    event::particle &event::particle::set_pz(double pz)
    {
        this->momentum_[3] = pz;
        return *this;
    }

    event::particle &event::particle::set_x(double x) { return this->set_px(x); }
    event::particle &event::particle::set_y(double y) { return this->set_py(y); }
    event::particle &event::particle::set_z(double z) { return this->set_pz(z); }
    event::particle &event::particle::set_t(double pt) { return this->set_E(pt); }

    event::particle &event::particle::set_mass(double m)
    {
        this->mass_ = m;
        return *this;
    }
    event::particle &event::particle::set_mUP(double m) { return this->set_mass(m); }
    event::particle &event::particle::set_m(double m) { return this->set_mass(m); }

    event::particle &event::particle::set_vtim(double v)
    {
        this->vtim_ = v;
        return *this;
    }
    event::particle &event::particle::set_vTimUP(double v) { return this->set_vtim(v); }

    event::particle &event::particle::set_spin(double s)
    {
        this->spin_ = s;
        return *this;
    }
    event::particle &event::particle::set_spinUP(double s) { return this->set_spin(s); }

    event::particle event::operator[](size_t i)
    {
        if (i >= size())
            throw std::out_of_range("event::operator[] index out of range");
        return get_particle(i);
    }

    event::particle event::at(size_t i)
    {
        if (i >= size())
            throw std::out_of_range("event::at index out of range");
        return get_particle(i);
    }

    event::const_particle event::operator[](size_t i) const
    {
        if (i >= size())
            throw std::out_of_range("event::operator[] index out of range");
        return get_particle(i);
    }

    event::const_particle event::at(size_t i) const
    {
        if (i >= size())
            throw std::out_of_range("event::at index out of range");
        return get_particle(i);
    }

    void event::validate() const
    {
        size_t s = this->n_; // number of partons in the event
        if (s == 0)
        {
            // If there are no particles, all vectors should be empty
            if (!momenta_.empty() || !mass_.empty() || !vtim_.empty() ||
                !spin_.empty() || !pdg_.empty() || !status_.empty() ||
                !mother_.empty() || !icol_.empty())
            {
                throw std::runtime_error("event::validate() failed: event has no particles, but vectors are not empty");
            }
            return; // Nothing to validate
        }
        auto check = [s](const auto &vec, const char *name)
        {
            if (vec.size() != s)
            {
                std::ostringstream oss;
                oss << "event::validate() failed: '" << name
                    << "' has size " << vec.size() << ", expected " << s;
                throw std::runtime_error(oss.str());
            }
        };

        check(momenta_, "momenta");
        check(mass_, "mass");
        check(vtim_, "vtim");
        check(spin_, "spin");
        check(pdg_, "pdg");
        check(status_, "status");
        check(mother_, "mother");
        check(icol_, "icol");
    }

    // Protected shared comparator function
    namespace
    {
        std::shared_ptr<cevent_equal_fn> event_equal_ptr =
            std::make_shared<cevent_equal_fn>(external_legs_const_comparator);
        std::mutex event_equal_mutex;
    }

    bool extra_fields_equal(const std::unordered_map<std::string, std::any> &a,
                            const std::unordered_map<std::string, std::any> &b)
    {
        if (a.size() != b.size())
            return false;

        for (const auto &[key, val_a] : a)
        {
            auto it = b.find(key);
            if (it == b.end())
                return false;

            const std::any &val_b = it->second;

            if (val_a.type() != val_b.type())
                return false;

            // Only compare known types
            if (val_a.type() == typeid(int))
            {
                if (std::any_cast<int>(val_a) != std::any_cast<int>(val_b))
                    return false;
            }
            else if (val_a.type() == typeid(double))
            {
                if (std::any_cast<double>(val_a) != std::any_cast<double>(val_b))
                    return false;
            }
            else if (val_a.type() == typeid(std::string))
            {
                if (std::any_cast<std::string>(val_a) != std::any_cast<std::string>(val_b))
                    return false;
            }
            else
            {
                warning("Unknown type for key: " + key + ", skipping (assuming equality)");
                return true; // Skip unknown types
            }
        }

        return true;
    }

    event &event::set_indices()
    {
        if (this->indices.empty())
        {
            this->indices.resize(this->size());
            std::iota(this->indices.begin(), this->indices.end(), 0);
        }
        return *this;
    }

    // Helper: key for (status, pdg)
    struct StatusPdgKey
    {
        short st;
        long pdg;
        bool operator==(const StatusPdgKey &o) const noexcept
        {
            return st == o.st && pdg == o.pdg;
        }
    };

    struct StatusPdgKeyHash
    {
        size_t operator()(const StatusPdgKey &k) const noexcept
        {
            // Cheap hash combine
            // Cast to unsigned to avoid UB on shifts of negative values.
            const auto a = static_cast<uint64_t>(static_cast<uint16_t>(k.st));
            const auto b = static_cast<uint64_t>(static_cast<uint64_t>(k.pdg));
            // 16 bits for status, rest for pdg; then mix
            uint64_t h = (a << 48) ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
            return static_cast<size_t>(h);
        }
    };

    event &event::set_indices(const event &other_event, bool fail_on_mismatch)
    {
        this->validate();
        other_event.validate();

        const size_t n_this = this->size();
        const size_t n_other = other_event.size();
        if (fail_on_mismatch && n_this != n_other)
        {
            throw std::runtime_error("event::set_indices: mismatched sizes");
        }

        // (status, pdg) -> queue of positions in other_event (in other’s order)
        std::unordered_map<StatusPdgKey, std::queue<size_t>, StatusPdgKeyHash> pos_by_key;
        pos_by_key.reserve(n_other * 2);
        for (size_t j = 0; j < n_other; ++j)
        {
            pos_by_key[StatusPdgKey{other_event.status_.at(j), other_event.pdg_.at(j)}].push(j);
        }

        // Fill indices so that indices[other_pos] = this_pos
        this->indices.assign(n_this, npos);
        for (size_t i = 0; i < n_this; ++i)
        {
            StatusPdgKey k{this->status_.at(i), this->pdg_.at(i)};
            auto it = pos_by_key.find(k);
            if (it == pos_by_key.end() || it->second.empty())
            {
                if (fail_on_mismatch)
                    throw std::runtime_error("event::set_indices: no remaining match for key");
                continue;
            }
            const size_t j = it->second.front();
            it->second.pop();
            this->indices[j] = i; // <-- the important line
        }
        // Optional: ensure none left unmatched and no npos remain
        if (fail_on_mismatch)
        {
            for (const auto &kv : pos_by_key)
            {
                if (!kv.second.empty())
                {
                    throw std::runtime_error("event::set_indices: surplus particles in other_event");
                }
            }
        }

        std::vector<size_t> override_indices;
        for (size_t k = 0; k < this->indices.size(); ++k)
        {
            if (fail_on_mismatch && this->indices[k] == npos)
            {
                throw std::runtime_error("event::set_indices: internal error, unfilled index");
            }
            if (this->indices[k] != npos)
            {
                override_indices.push_back(this->indices[k]);
            }
        }
        this->set_indices(override_indices);
        return *this;
    }

    event &event::set_indices(const std::vector<size_t> &idxs)
    {
        if (idxs.size() > this->size())
        {
            throw std::runtime_error("event::set_indices: size mismatch");
        }
        this->indices = idxs;
        return *this;
    }

    void event::print_extra(std::ostream &os) const
    {
        for (const auto &[key, val] : this->extra)
        {
            if (val.type() == typeid(int))
            {
                os << "<" << key << "\">" << std::any_cast<int>(val) << "</" << key << ">\n";
            }
            else if (val.type() == typeid(double))
            {
                os << std::setprecision(10) << std::scientific;
                os << "<" << key << "\">" << std::any_cast<double>(val) << "</" << key << ">\n";
            }
            else if (val.type() == typeid(std::string))
            {
                os << "<" << key << "\">" << std::any_cast<std::string>(val) << "</" << key << ">\n";
            }
            else if (val.type() == typeid(std::string_view))
            {
                os << "<" << key << "\">" << std::any_cast<std::string_view>(val) << "</" << key << ">\n";
            }
            else if (val.type() == typeid(xmlNode))
            {
                std::any_cast<xmlNode>(val).write(os);
                os << "\n";
            }
            else if (val.type() == typeid(std::shared_ptr<xmlNode>))
            {
                std::any_cast<std::shared_ptr<xmlNode>>(val)->write(os);
                os << "\n";
            }
            else
            {
                warning("Unknown type for extra field: " + key + ", skipping print");
            }
        }
    }

    void event::print_scales(std::ostream &os) const
    {
        bool scales = false;
        std::string scale_str = "<scales";
        if (this->muF_ != 0.0 && (this->scale_ - this->muF_) / this->scale_ > 1e-6)
        {
            scales = true;
            scale_str += " muf=\'" + std::to_string(this->muF_) + "\'";
        }
        if (this->muR_ != 0.0 && (this->scale_ - this->muR_) / this->scale_ > 1e-6)
        {
            scales = true;
            scale_str += " mur=\'" + std::to_string(this->muR_) + "\'";
        }
        if (this->muPS_ != 0.0 && (this->scale_ - this->muPS_) / this->scale_ > 1e-6)
        {
            scales = true;
            scale_str += " muPS=\'" + std::to_string(this->muPS_) + "\'";
        }
        if (scales)
        {
            scale_str += ">\n";
            os << scale_str;
        }
    }

    void event::print(std::ostream &os, bool include_ids) const
    {
        os << "\n<event>\n";
        this->print_head(os);
        for (auto prt : *this)
        {
            prt.print(os);
        }
        this->print_scales(os);
        this->print_extra(os);
        this->print_wgts(os, include_ids);
        os << "</event>";
    }

    bool default_event_equal(const event &lhs, const event &rhs)
    {
        return lhs.momenta_ == rhs.momenta_ &&
               lhs.mass_ == rhs.mass_ &&
               lhs.vtim_ == rhs.vtim_ &&
               lhs.spin_ == rhs.spin_ &&
               lhs.pdg_ == rhs.pdg_ &&
               lhs.status_ == rhs.status_ &&
               lhs.mother_ == rhs.mother_ &&
               lhs.icol_ == rhs.icol_ &&
               lhs.n_ == rhs.n_ &&
               lhs.proc_id_ == rhs.proc_id_ &&
               lhs.weight_ == rhs.weight_ &&
               lhs.scale_ == rhs.scale_ &&
               lhs.alphaEW_ == rhs.alphaEW_ &&
               lhs.alphaS_ == rhs.alphaS_ &&
               extra_fields_equal(lhs.extra, rhs.extra);
    }

    bool operator==(const event &lhs, const event &rhs)
    {
        std::shared_ptr<cevent_equal_fn> fn;
        {
            std::lock_guard<std::mutex> lock(event_equal_mutex);
            fn = event_equal_ptr;
        }
        return (*fn)(lhs, rhs);
    }

    bool operator!=(const event &lhs, const event &rhs)
    {
        return !(lhs == rhs);
    }

    void set_event_comparator(cevent_equal_fn fn)
    {
        std::lock_guard<std::mutex> lock(event_equal_mutex);
        event_equal_ptr = std::make_shared<cevent_equal_fn>(std::move(fn));
    }

    void reset_event_comparator()
    {
        set_event_comparator(external_legs_const_comparator);
    }

    bool external_legs_comparator(event &a, event &b)
    {
        auto count_pdgs = [](const event &e, int status_filter)
        {
            std::unordered_map<int, int> pdg_counts;
            for (size_t i = 0; i < e.size(); ++i)
            {
                if (e.status_[i] == status_filter)
                {
                    ++pdg_counts[e.pdg_[i]];
                }
            }
            return pdg_counts;
        };

        // Count initial and final particles separately
        auto init_a = count_pdgs(a, -1);
        auto init_b = count_pdgs(b, -1);
        auto final_a = count_pdgs(a, +1);
        auto final_b = count_pdgs(b, +1);

        return init_a == init_b && final_a == final_b;
    }

    bool external_legs_const_comparator(const event &a, const event &b)
    {
        auto count_pdgs = [](const event &e, int status_filter)
        {
            std::unordered_map<int, int> pdg_counts;
            for (size_t i = 0; i < e.size(); ++i)
            {
                if (e.status_[i] == status_filter)
                {
                    ++pdg_counts[e.pdg_[i]];
                }
            }
            return pdg_counts;
        };

        // Count initial and final particles separately
        auto init_a = count_pdgs(a, -1);
        auto init_b = count_pdgs(b, -1);
        auto final_a = count_pdgs(a, +1);
        auto final_b = count_pdgs(b, +1);

        return init_a == init_b && final_a == final_b;
    }

    bool always_true(const event &a, const event &b)
    {
        UNUSED(a);
        UNUSED(b);
        return true;
    }

    struct particleKey
    {
        double mass = 0.0, vtim = 0.0, spin = 0.0;
        arr4<double> momentum{};
        long int pdg = 0;
        short int status = 0;
        arr2<short int> mother{}, icol{};

        bool operator==(const particleKey &other) const
        {
            return mass == other.mass && vtim == other.vtim && spin == other.spin &&
                   momentum == other.momentum && pdg == other.pdg && status == other.status &&
                   mother == other.mother && icol == other.icol;
        }

        bool operator<(const particleKey &other) const
        {
            return std::tie(mass, vtim, spin, momentum, pdg, status, mother, icol) <
                   std::tie(other.mass, other.vtim, other.spin, other.momentum,
                            other.pdg, other.status, other.mother, other.icol);
        }
        std::tuple<
            std::optional<double>,
            std::optional<double>,
            std::optional<double>,
            std::array<std::optional<double>, 4>,
            std::optional<long int>,
            std::optional<short int>,
            std::optional<arr2<short int>>,
            std::optional<arr2<short int>>>
        sort_key(const eventComparatorConfig &cfg) const
        {
            return {
                cfg.compare_mass ? std::make_optional(mass) : std::nullopt,
                cfg.compare_vtim ? std::make_optional(vtim) : std::nullopt,
                cfg.compare_spin ? std::make_optional(spin) : std::nullopt,
                {cfg.compare_momentum_E ? std::make_optional(momentum[0]) : std::nullopt,
                 cfg.compare_momentum_x ? std::make_optional(momentum[1]) : std::nullopt,
                 cfg.compare_momentum_y ? std::make_optional(momentum[2]) : std::nullopt,
                 cfg.compare_momentum_z ? std::make_optional(momentum[3]) : std::nullopt},
                cfg.compare_pdg ? std::make_optional(pdg) : std::nullopt,
                cfg.compare_status ? std::make_optional(status) : std::nullopt,
                cfg.compare_mother ? std::make_optional(mother) : std::nullopt,
                cfg.compare_icol ? std::make_optional(icol) : std::nullopt};
        }
    };

    inline bool nearly_equal_rel(double a, double b, double tol)
    {
        if (a == b)
            return true;
        double denom = std::abs(a) + std::abs(b);
        return denom == 0.0 ? false : std::abs(a - b) / denom < tol;
    }

    event_equal_fn eventComparatorConfig::make_comparator() const
    {
        return [*this](const event &a, const event &b) -> bool
        {
            auto extract_keys = [&](const event &ev)
            {
                std::vector<particleKey> keys;
                for (size_t i = 0; i < ev.size(); ++i)
                {
                    if (!status_filter.empty() && !status_filter.count(ev.status_[i]))
                        continue;

                    particleKey key;
                    if (compare_mass)
                        key.mass = ev.mass_[i];
                    if (compare_vtim)
                        key.vtim = ev.vtim_[i];
                    if (compare_spin)
                        key.spin = ev.spin_[i];

                    if (compare_momentum)
                    {
                        key.momentum = {
                            compare_momentum_E ? ev.momenta_[i][0] : 0.0,
                            compare_momentum_x ? ev.momenta_[i][1] : 0.0,
                            compare_momentum_y ? ev.momenta_[i][2] : 0.0,
                            compare_momentum_z ? ev.momenta_[i][3] : 0.0};
                    }

                    if (compare_pdg)
                        key.pdg = ev.pdg_[i];
                    if (compare_status)
                        key.status = ev.status_[i];
                    if (compare_mother)
                        key.mother = ev.mother_[i];
                    if (compare_icol)
                        key.icol = ev.icol_[i];

                    keys.push_back(std::move(key));
                }
                std::sort(keys.begin(), keys.end(), [&](const particleKey &a, const particleKey &b)
                          { return a.sort_key(*this) < b.sort_key(*this); });
                return keys;
            };

            bool local_compare_n = compare_n;
            if (!status_filter.empty() && compare_n)
            {
                warning("compare_n=true with status_filter active — ignoring compare_n.");
                // Force-disable to prevent false mismatches
                local_compare_n = false;
            }

            auto a_keys = extract_keys(a);
            auto b_keys = extract_keys(b);
            if (a_keys.size() != b_keys.size())
                return false;

            for (size_t i = 0; i < a_keys.size(); ++i)
            {
                const auto &ka = a_keys[i];
                const auto &kb = b_keys[i];
                if (compare_mass && !nearly_equal_rel(ka.mass, kb.mass, mass_tol))
                    return false;
                if (compare_vtim && !nearly_equal_rel(ka.vtim, kb.vtim, vtim_tol))
                    return false;
                if (compare_spin && !nearly_equal_rel(ka.spin, kb.spin, spin_tol))
                    return false;
                if (compare_momentum)
                {
                    if (compare_momentum_E && !nearly_equal_rel(ka.momentum[0], kb.momentum[0], momentum_tol))
                        return false;
                    if (compare_momentum_x && !nearly_equal_rel(ka.momentum[1], kb.momentum[1], momentum_tol))
                        return false;
                    if (compare_momentum_y && !nearly_equal_rel(ka.momentum[2], kb.momentum[2], momentum_tol))
                        return false;
                    if (compare_momentum_z && !nearly_equal_rel(ka.momentum[3], kb.momentum[3], momentum_tol))
                        return false;
                }

                if (compare_pdg && ka.pdg != kb.pdg)
                    return false;
                if (compare_status && ka.status != kb.status)
                    return false;
                if (compare_mother && ka.mother != kb.mother)
                    return false;
                if (compare_icol && ka.icol != kb.icol)
                    return false;
            }
            if (local_compare_n && a.n_ != b.n_)
                return false;
            if (compare_proc_id && a.proc_id_ != b.proc_id_)
                return false;
            if (compare_weight && !nearly_equal_rel(a.weight_, b.weight_, weight_tol))
                return false;
            if (compare_scale && !nearly_equal_rel(a.scale_, b.scale_, scale_tol))
                return false;
            if (compare_alphaEW && !nearly_equal_rel(a.alphaEW_, b.alphaEW_, alphaEW_tol))
                return false;
            if (compare_alphaS && !nearly_equal_rel(a.alphaS_, b.alphaS_, alphaS_tol))
                return false;
            return true;
        };
    }

    cevent_equal_fn eventComparatorConfig::make_const_comparator() const
    {
        return [*this](const event &a, const event &b) -> bool
        {
            auto extract_keys = [&](const event &ev)
            {
                std::vector<particleKey> keys;
                for (size_t i = 0; i < ev.size(); ++i)
                {
                    if (!status_filter.empty() && !status_filter.count(ev.status_[i]))
                        continue;

                    particleKey key;
                    if (compare_mass)
                        key.mass = ev.mass_[i];
                    if (compare_vtim)
                        key.vtim = ev.vtim_[i];
                    if (compare_spin)
                        key.spin = ev.spin_[i];

                    if (compare_momentum)
                    {
                        key.momentum = {
                            compare_momentum_E ? ev.momenta_[i][0] : 0.0,
                            compare_momentum_x ? ev.momenta_[i][1] : 0.0,
                            compare_momentum_y ? ev.momenta_[i][2] : 0.0,
                            compare_momentum_z ? ev.momenta_[i][3] : 0.0};
                    }

                    if (compare_pdg)
                        key.pdg = ev.pdg_[i];
                    if (compare_status)
                        key.status = ev.status_[i];
                    if (compare_mother)
                        key.mother = ev.mother_[i];
                    if (compare_icol)
                        key.icol = ev.icol_[i];

                    keys.push_back(std::move(key));
                }
                std::sort(keys.begin(), keys.end(), [&](const particleKey &a, const particleKey &b)
                          { return a.sort_key(*this) < b.sort_key(*this); });
                return keys;
            };

            bool local_compare_n = compare_n;
            if (!status_filter.empty() && compare_n)
            {
                warning("compare_n=true with status_filter active — ignoring compare_n.");
                // Force-disable to prevent false mismatches
                local_compare_n = false;
            }

            auto a_keys = extract_keys(a);
            auto b_keys = extract_keys(b);
            if (a_keys.size() != b_keys.size())
                return false;

            for (size_t i = 0; i < a_keys.size(); ++i)
            {
                const auto &ka = a_keys[i];
                const auto &kb = b_keys[i];
                if (compare_mass && !nearly_equal_rel(ka.mass, kb.mass, mass_tol))
                    return false;
                if (compare_vtim && !nearly_equal_rel(ka.vtim, kb.vtim, vtim_tol))
                    return false;
                if (compare_spin && !nearly_equal_rel(ka.spin, kb.spin, spin_tol))
                    return false;
                if (compare_momentum)
                {
                    if (compare_momentum_E && !nearly_equal_rel(ka.momentum[0], kb.momentum[0], momentum_tol))
                        return false;
                    if (compare_momentum_x && !nearly_equal_rel(ka.momentum[1], kb.momentum[1], momentum_tol))
                        return false;
                    if (compare_momentum_y && !nearly_equal_rel(ka.momentum[2], kb.momentum[2], momentum_tol))
                        return false;
                    if (compare_momentum_z && !nearly_equal_rel(ka.momentum[3], kb.momentum[3], momentum_tol))
                        return false;
                }

                if (compare_pdg && ka.pdg != kb.pdg)
                    return false;
                if (compare_status && ka.status != kb.status)
                    return false;
                if (compare_mother && ka.mother != kb.mother)
                    return false;
                if (compare_icol && ka.icol != kb.icol)
                    return false;
            }
            if (local_compare_n && a.n_ != b.n_)
                return false;
            if (compare_proc_id && a.proc_id_ != b.proc_id_)
                return false;
            if (compare_weight && !nearly_equal_rel(a.weight_, b.weight_, weight_tol))
                return false;
            if (compare_scale && !nearly_equal_rel(a.scale_, b.scale_, scale_tol))
                return false;
            if (compare_alphaEW && !nearly_equal_rel(a.alphaEW_, b.alphaEW_, alphaEW_tol))
                return false;
            if (compare_alphaS && !nearly_equal_rel(a.alphaS_, b.alphaS_, alphaS_tol))
                return false;
            return true;
        };
    }

    eventComparatorConfig compare_legs_only()
    {
        return eventComparatorConfig{}
            .set_mass(false)
            .set_pdg(true)
            .set_status(false)
            .set_spin(false)
            .set_vtim(false)
            .set_momentum(false)
            .set_status_filter(std::vector<int>{+1, -1})
            .set_n(false)
            .set_proc_id(false)
            .set_weight(false)
            .set_scale(false)
            .set_alphaEW(false)
            .set_alphaS(false)
            .set_mother(false)
            .set_icol(false);
    }

    eventComparatorConfig compare_final_state_only()
    {
        return eventComparatorConfig{}
            .set_mass(true)
            .set_pdg(true)
            .set_momentum(true)
            .set_status_filter(+1)
            .set_vtim(true)
            .set_spin(true)
            .set_mother(true)
            .set_icol(true);
    }

    eventComparatorConfig compare_physics_fields()
    {
        return eventComparatorConfig{}
            .set_mass(true)
            .set_pdg(true)
            .set_status(true)
            .set_spin(true)
            .set_vtim(true)
            .set_momentum(true)
            .set_n(true);
    }

    eventBelongs::eventBelongs(const event &e)
    {
        this->events.push_back(std::make_shared<event>(e));
        this->comparator = external_legs_comparator;
        this->const_comparator = external_legs_const_comparator;
    }

    eventBelongs::eventBelongs(std::shared_ptr<event> e)
    {
        this->events.push_back(e);
        this->comparator = external_legs_comparator;
        this->const_comparator = external_legs_const_comparator;
    }

    eventBelongs::eventBelongs(std::vector<event> evs)
    {
        this->events = {};
        for (const auto &e : evs)
        {
            this->events.push_back(std::make_shared<event>(e));
        }
        this->comparator = external_legs_comparator;
        this->const_comparator = external_legs_const_comparator;
    }

    eventBelongs::eventBelongs(std::vector<std::shared_ptr<event>> evs)
    {
        this->events = evs;
        this->comparator = external_legs_comparator;
        this->const_comparator = external_legs_const_comparator;
    }

    eventBelongs::eventBelongs(const event &e, event_equal_fn comp)
    {
        this->events.push_back(std::make_shared<event>(e));
        this->comparator = std::move(comp);
    }

    eventBelongs::eventBelongs(std::shared_ptr<event> e, event_equal_fn comp)
    {
        this->events.push_back(e);
        this->comparator = std::move(comp);
    }

    eventBelongs::eventBelongs(std::vector<event> evs, event_equal_fn comp)
    {
        this->events = {};
        for (const auto &e : evs)
        {
            this->events.push_back(std::make_shared<event>(e));
        }
        this->comparator = std::move(comp);
    }

    eventBelongs::eventBelongs(std::vector<std::shared_ptr<event>> evs, event_equal_fn comp)
    {
        this->events = evs;
        this->comparator = std::move(comp);
    }

    eventBelongs::eventBelongs(const event &e, cevent_equal_fn comp)
    {
        this->events.push_back(std::make_shared<event>(e));
        this->comparator = comp;
        this->const_comparator = std::move(comp);
    }

    eventBelongs::eventBelongs(std::shared_ptr<event> e, cevent_equal_fn comp)
    {
        this->events.push_back(e);
        this->comparator = comp;
        this->const_comparator = std::move(comp);
    }

    eventBelongs::eventBelongs(std::vector<event> evs, cevent_equal_fn comp)
    {
        this->events = {};
        for (const auto &e : evs)
        {
            this->events.push_back(std::make_shared<event>(e));
        }
        this->comparator = comp;
        this->const_comparator = std::move(comp);
    }

    eventBelongs::eventBelongs(std::vector<std::shared_ptr<event>> evs, cevent_equal_fn comp)
    {
        this->events = evs;
        this->comparator = comp;
        this->const_comparator = std::move(comp);
    }

    eventBelongs::eventBelongs(const event &e, event_equal_fn comp, cevent_equal_fn ccomp)
    {
        this->events.push_back(std::make_shared<event>(e));
        this->comparator = std::move(comp);
        this->const_comparator = std::move(ccomp);
    }

    eventBelongs::eventBelongs(std::shared_ptr<event> e, event_equal_fn comp, cevent_equal_fn ccomp)
    {
        this->events.push_back(e);
        this->comparator = std::move(comp);
        this->const_comparator = std::move(ccomp);
    }

    eventBelongs::eventBelongs(std::vector<event> evs, event_equal_fn comp, cevent_equal_fn ccomp)
    {
        this->events = {};
        for (const auto &e : evs)
        {
            this->events.push_back(std::make_shared<event>(e));
        }
        this->comparator = std::move(comp);
        this->const_comparator = std::move(ccomp);
    }

    eventBelongs::eventBelongs(std::vector<std::shared_ptr<event>> evs, event_equal_fn comp, cevent_equal_fn ccomp)
    {
        this->events = evs;
        this->comparator = std::move(comp);
        this->const_comparator = std::move(ccomp);
    }

    eventBelongs &eventBelongs::add_event(const event &e)
    {
        this->events.push_back(std::make_shared<event>(e));
        return *this;
    }

    eventBelongs &eventBelongs::add_event(std::shared_ptr<event> e)
    {
        this->events.push_back(e);
        return *this;
    }

    eventBelongs &eventBelongs::add_event(const std::vector<event> &evs)
    {
        for (const auto &e : evs)
        {
            this->events.push_back(std::make_shared<event>(e));
        }
        return *this;
    }

    eventBelongs &eventBelongs::add_event(std::vector<std::shared_ptr<event>> evs)
    {
        for (const auto &e : evs)
        {
            this->events.push_back(e);
        }
        return *this;
    }

    eventBelongs &eventBelongs::set_events(const event &e)
    {
        this->events.clear();
        this->events.push_back(std::make_shared<event>(e));
        return *this;
    }

    eventBelongs &eventBelongs::set_events(std::shared_ptr<event> e)
    {
        this->events.clear();
        this->events.push_back(e);
        return *this;
    }

    eventBelongs &eventBelongs::set_events(const std::vector<event> &evs)
    {
        this->events = {};
        for (const auto &e : evs)
        {
            this->events.push_back(std::make_shared<event>(e));
        }
        return *this;
    }

    eventBelongs &eventBelongs::set_events(std::vector<std::shared_ptr<event>> evs)
    {
        this->events = evs;
        return *this;
    }

    eventBelongs &eventBelongs::set_comparator(event_equal_fn comp)
    {
        this->comparator = std::move(comp);
        return *this;
    }

    eventBelongs &eventBelongs::set_comparator(cevent_equal_fn comp)
    {
        this->comparator = comp;
        this->const_comparator = std::move(comp);
        return *this;
    }

    eventBelongs &eventBelongs::set_comparator(const eventComparatorConfig &cfg)
    {
        this->comparator = cfg.make_comparator();
        this->const_comparator = cfg.make_const_comparator();
        return *this;
    }

    // If non-const and an event belongs, set its indices automatically
    bool eventBelongs::belongs_mutable(event &e)
    {
        if (this->events.empty() && !(this->comparator || this->const_comparator))
        {
            throw std::runtime_error("eventBelongs::belongs() called with no events set");
        }
        if (!this->comparator)
        {
            if (!this->const_comparator)
                throw std::runtime_error("eventBelongs::belongs() called with no comparator set");
            // Fallback to const comparator if available
            return this->belongs_const(static_cast<const event &>(e));
        }
        for (auto &ev : this->events)
        {
            if (this->comparator(*ev, e))
            {
                e.set_indices(*ev);
                return true;
            }
        }
        return false;
    }

    bool eventBelongs::belongs_const(const event &e) const
    {
        if (this->events.empty())
        {
            throw std::runtime_error("eventBelongs::belongs() called with no events set");
        }
        if (!this->const_comparator)
        {
            throw std::runtime_error("eventBelongs::belongs() called with no const_comparator set");
        }
        for (const auto &ev : this->events)
        {
            if (this->const_comparator(*ev, e))
            {
                return true;
            }
        }
        return false;
    }

    bool eventBelongs::belongs(event &e)
    {
        return this->belongs_mutable(e);
    }

    bool eventBelongs::belongs(const event &e) const
    {
        return this->belongs_const(e);
    }

    bool eventBelongs::belongs(std::shared_ptr<event> e)
    {
        return this->belongs_mutable(*e);
    }

    event_bool_fn eventBelongs::get_event_bool()
    {
        return [this](event &e) -> bool
        {
            return this->belongs_mutable(e);
        };
    }

    cevent_bool_fn eventBelongs::get_const_event_bool() const
    {
        return [this](const event &e) -> bool
        {
            return this->belongs_const(e);
        };
    }

    eventBelongs all_events_belong()
    {
        event e1(0);
        return eventBelongs(e1, always_true, always_true);
    }

    eventSorter::eventSorter(const eventBelongs &e_set)
    {
        this->event_sets.push_back(std::make_shared<eventBelongs>(e_set));
        this->comparators.push_back(this->event_sets.back()->get_event_bool());
        this->const_comparators.push_back(this->event_sets.back()->get_const_event_bool());
    }

    eventSorter::eventSorter(event_bool_fn comp)
    {
        this->comparators.push_back(std::move(comp));
    }

    eventSorter::eventSorter(cevent_bool_fn comp)
    {
        this->comparators.push_back([comp](event &e)
                                    { return comp(e); });
        this->const_comparators.push_back(std::move(comp));
    }

    eventSorter::eventSorter(event_bool_fn comp, cevent_bool_fn ccomp)
    {
        this->comparators.push_back(std::move(comp));
        this->const_comparators.push_back(std::move(ccomp));
    }

    eventSorter::eventSorter(std::vector<eventBelongs> e_sets)
    {
        this->event_sets = {};
        for (const auto &es : e_sets)
        {
            this->event_sets.push_back(std::make_shared<eventBelongs>(es));
            this->comparators.push_back(this->event_sets.back()->get_event_bool());
            this->const_comparators.push_back(this->event_sets.back()->get_const_event_bool());
        }
    }

    eventSorter::eventSorter(std::vector<event_bool_fn> comps)
    {
        this->comparators = std::move(comps);
    }

    eventSorter::eventSorter(std::vector<event_bool_fn> comps, std::vector<cevent_bool_fn> ccomps)
    {
        if (comps.size() != ccomps.size())
        {
            throw std::runtime_error("eventSorter: size mismatch in constructor");
        }
        this->comparators = std::move(comps);
        this->const_comparators = std::move(ccomps);
    }

    eventSorter &eventSorter::add_event_set(const eventBelongs &e_set)
    {
        this->event_sets.push_back(std::make_shared<eventBelongs>(e_set));
        this->comparators.push_back(this->event_sets.back()->get_event_bool());
        this->const_comparators.push_back(this->event_sets.back()->get_const_event_bool());
        return *this;
    }

    eventSorter &eventSorter::add_event_set(const std::vector<eventBelongs> &e_sets)
    {
        size_t old_size = this->event_sets.size();
        for (const auto &es : e_sets)
        {
            this->event_sets.push_back(std::make_shared<eventBelongs>(es));
        }
        this->comparators.reserve(this->event_sets.size());
        this->const_comparators.reserve(this->event_sets.size());
        for (size_t i = old_size; i < this->event_sets.size(); ++i)
        {
            auto &es = this->event_sets[i];
            this->comparators.push_back(es->get_event_bool());
            this->const_comparators.push_back(es->get_const_event_bool());
        }
        return *this;
    }

    eventSorter &eventSorter::add_bool(event_bool_fn comp)
    {
        this->comparators.push_back(std::move(comp));
        return *this;
    }

    eventSorter &eventSorter::add_const_bool(cevent_bool_fn comp)
    {
        this->comparators.push_back([comp](event &e)
                                    { return comp(e); });
        this->const_comparators.push_back(std::move(comp));
        return *this;
    }

    eventSorter &eventSorter::add_bool(event_bool_fn comp, cevent_bool_fn ccomp)
    {
        this->comparators.push_back(std::move(comp));
        this->const_comparators.push_back(std::move(ccomp));
        return *this;
    }

    eventSorter &eventSorter::add_bool(std::vector<event_bool_fn> comps)
    {
        this->comparators.insert(this->comparators.end(), std::make_move_iterator(comps.begin()), std::make_move_iterator(comps.end()));
        return *this;
    }

    eventSorter &eventSorter::add_const_bool(std::vector<cevent_bool_fn> ccomps)
    {
        for (auto &c : ccomps)
        {
            this->comparators.push_back([c](event &e)
                                        { return c(e); });
        }
        this->const_comparators.insert(this->const_comparators.end(), std::make_move_iterator(ccomps.begin()), std::make_move_iterator(ccomps.end()));
        return *this;
    }

    eventSorter &eventSorter::add_bool(std::vector<event_bool_fn> comps, std::vector<cevent_bool_fn> ccomps)
    {
        if (comps.size() != ccomps.size())
        {
            throw std::runtime_error("eventSorter::add_bool: size mismatch");
        }
        this->comparators.insert(this->comparators.end(), std::make_move_iterator(comps.begin()), std::make_move_iterator(comps.end()));
        this->const_comparators.insert(this->const_comparators.end(), std::make_move_iterator(ccomps.begin()), std::make_move_iterator(ccomps.end()));
        return *this;
    }

    eventSorter &eventSorter::set_event_sets(const eventBelongs &e_set)
    {
        this->event_sets.clear();
        this->event_sets.push_back(std::make_shared<eventBelongs>(e_set));
        this->comparators.clear();
        this->const_comparators.clear();
        this->comparators.push_back(this->event_sets.back()->get_event_bool());
        this->const_comparators.push_back(this->event_sets.back()->get_const_event_bool());
        return *this;
    }

    eventSorter &eventSorter::set_event_sets(const std::vector<eventBelongs> &e_sets)
    {
        this->event_sets = {};
        this->comparators.clear();
        this->const_comparators.clear();
        this->event_sets.reserve(e_sets.size());
        for (const auto &es : e_sets)
        {
            this->event_sets.push_back(std::make_shared<eventBelongs>(es));
            this->comparators.push_back(this->event_sets.back()->get_event_bool());
            this->const_comparators.push_back(this->event_sets.back()->get_const_event_bool());
        }
        return *this;
    }

    eventSorter &eventSorter::set_bools(event_bool_fn comp)
    {
        this->event_sets.clear();
        this->comparators = {std::move(comp)};
        this->const_comparators.clear();
        return *this;
    }

    eventSorter &eventSorter::set_const_bools(cevent_bool_fn comp)
    {
        this->event_sets.clear();
        this->comparators = {[comp](event &e)
                             { return comp(e); }};
        this->const_comparators = {std::move(comp)};
        return *this;
    }

    eventSorter &eventSorter::set_bools(event_bool_fn comp, cevent_bool_fn ccomp)
    {
        this->event_sets.clear();
        this->comparators = {std::move(comp)};
        this->const_comparators = {std::move(ccomp)};
        return *this;
    }

    eventSorter &eventSorter::set_bools(std::vector<event_bool_fn> comps)
    {
        this->event_sets.clear();
        this->comparators = std::move(comps);
        this->const_comparators.clear();
        return *this;
    }

    eventSorter &eventSorter::set_const_bools(std::vector<cevent_bool_fn> ccomps)
    {
        this->event_sets.clear();
        this->comparators = {};
        for (const auto &c : ccomps)
        {
            this->comparators.push_back([c](event &e)
                                        { return c(e); });
        }
        this->const_comparators = std::move(ccomps);
        return *this;
    }

    eventSorter &eventSorter::set_bools(std::vector<event_bool_fn> comps, std::vector<cevent_bool_fn> ccomps)
    {
        if (comps.size() != ccomps.size())
        {
            throw std::runtime_error("eventSorter::set_bools: size mismatch");
        }
        this->event_sets.clear();
        this->comparators = std::move(comps);
        this->const_comparators = std::move(ccomps);
        return *this;
    }

    size_t eventSorter::size() const
    {
        if (this->comparators.size() != this->const_comparators.size() && this->const_comparators.size() != 0)
        {
            throw std::runtime_error("eventSorter::size(): Inconsistent internal state in eventSorter");
        }
        return std::max(this->comparators.size(), this->event_sets.size());
    }

    void eventSorter::extract_comparators()
    {
        this->comparators.clear();
        this->const_comparators.clear();
        for (auto &es : this->event_sets)
        {
            this->comparators.push_back(es->get_event_bool());
            this->const_comparators.push_back(es->get_const_event_bool());
        }
    }

    size_t eventSorter::position(event &e)
    {
        if (this->comparators.size() < this->event_sets.size())
            this->extract_comparators();

        for (size_t i = 0; i < this->comparators.size(); ++i)
        {
            if (!this->comparators[i])
                continue;
            if (this->comparators[i](e))
                return i;
        }
        return npos;
    }

    size_t eventSorter::position(const event &e) const
    {
        if (this->const_comparators.size() < this->event_sets.size())
            const_cast<eventSorter *>(this)->extract_comparators();

        for (size_t i = 0; i < this->const_comparators.size(); ++i)
        {
            if (!this->const_comparators[i])
                continue;
            if (this->const_comparators[i](e))
                return i;
        }
        return npos;
    }

    size_t eventSorter::position(std::shared_ptr<event> e)
    {
        if (this->comparators.size() < this->event_sets.size())
        {
            this->extract_comparators();
        }
        for (size_t i = 0; i < this->comparators.size(); ++i)
        {
            if (this->comparators[i](*e))
            {
                return i;
            }
        }
        return npos;
    }

    std::vector<size_t> eventSorter::position(std::vector<event> &evs)
    {
        std::vector<size_t> positions(evs.size(), npos);
        for (size_t i = 0; i < evs.size(); ++i)
        {
            positions[i] = this->position(evs[i]);
        }
        return positions;
    }

    std::vector<size_t> eventSorter::position(const std::vector<event> &evs) const
    {
        std::vector<size_t> positions(evs.size(), npos);
        for (size_t i = 0; i < evs.size(); ++i)
        {
            positions[i] = this->position(evs[i]);
        }
        return positions;
    }

    std::vector<size_t> eventSorter::position(std::vector<std::shared_ptr<event>> evs)
    {
        std::vector<size_t> positions;
        for (auto e : evs)
        {
            positions.push_back(this->position(e));
        }
        return positions;
    }

    std::vector<size_t> eventSorter::sort(std::vector<event> &evs)
    {
        return this->position(evs);
    }

    std::vector<size_t> eventSorter::sort(const std::vector<event> &evs) const
    {
        return this->position(evs);
    }

    std::vector<size_t> eventSorter::sort(std::vector<std::shared_ptr<event>> evts)
    {
        return this->position(evts);
    }

    event_hash_fn eventSorter::get_hash()
    {
        return [this](event &e) -> size_t
        {
            size_t pos = this->position(e);
            return pos;
        };
    }

    cevent_hash_fn eventSorter::get_const_hash() const
    {
        return [this](const event &e) -> size_t
        {
            size_t pos = this->position(e);
            return pos;
        };
    }

    eventSorter make_sample_sorter(const std::vector<event> &sample, event_equal_fn comp)
    {
        if (sample.empty())
        {
            throw std::invalid_argument("Sample vector is empty");
        }
        eventSorter sorter;
        for (const auto &ev : sample)
        {
            if (sorter.position(ev) == npos)
            {
                sorter.add_event_set(eventBelongs{ev, comp});
            }
        }
        return sorter;
    }

    eventSorter make_sample_sorter(std::vector<std::shared_ptr<event>> sample, event_equal_fn comp)
    {
        if (sample.empty())
        {
            throw std::invalid_argument("Sample vector is empty");
        }
        eventSorter sorter;
        for (auto ev : sample)
        {
            if (sorter.position(ev) == npos)
            {
                sorter.add_event_set(eventBelongs{*ev, comp});
            }
        }
        return sorter;
    }

    process::process(std::vector<std::shared_ptr<event>> evs, bool filter_partons)
    {
        this->filter = filter_partons;
        this->add_event(evs);
    }

    process::process(std::vector<event> evs, bool filter_partons)
    {
        this->filter = filter_partons;
        this->add_event(evs);
    }

    process &process::add_event_raw(const event &ev)
    {
        auto summed_n = (this->n_summed.empty() ? 0 : this->n_summed.back()) + ev.n_;
        this->n_.push_back(ev.n_);
        this->n_summed.push_back(summed_n);
        this->proc_id_.push_back(ev.proc_id_);
        this->weight_.push_back(ev.weight_);
        this->scale_.push_back(ev.scale_);
        auto muF = (ev.muF_ == 0.0) ? ev.scale_ : ev.muF_;
        auto muR = (ev.muR_ == 0.0) ? ev.scale_ : ev.muR_;
        auto muPS = (ev.muPS_ == 0.0) ? ev.scale_ : ev.muPS_;
        this->muF_.push_back(muF);
        this->muR_.push_back(muR);
        this->muPS_.push_back(muPS);
        this->alphaEW_.push_back(ev.alphaEW_);
        this->alphaS_.push_back(ev.alphaS_);
        for (auto prtcl : ev)
        {
            this->momenta_.push_back(prtcl.momentum_);
            this->mass_.push_back(prtcl.mass_);
            this->vtim_.push_back(prtcl.vtim_);
            this->spin_.push_back(prtcl.spin_);
            this->pdg_.push_back(prtcl.pdg_);
            this->status_.push_back(prtcl.status_);
            this->mother_.push_back(prtcl.mother_);
            this->icol_.push_back(prtcl.icol_);
        }
        this->wgts_.push_back(ev.wgts_);
        this->add_extra(ev.extra);
        this->events.push_back(std::make_shared<event>(ev));
        return *this;
    }

    process &process::add_event_raw(std::shared_ptr<event> ev)
    {
        auto summed_n = (this->n_summed.empty() ? 0 : this->n_summed.back()) + ev->n_;
        this->n_.push_back(ev->n_);
        this->n_summed.push_back(summed_n);
        this->proc_id_.push_back(ev->proc_id_);
        this->weight_.push_back(ev->weight_);
        this->scale_.push_back(ev->scale_);
        auto muF = (ev->muF_ == 0.0) ? ev->scale_ : ev->muF_;
        auto muR = (ev->muR_ == 0.0) ? ev->scale_ : ev->muR_;
        auto muPS = (ev->muPS_ == 0.0) ? ev->scale_ : ev->muPS_;
        this->muF_.push_back(muF);
        this->muR_.push_back(muR);
        this->muPS_.push_back(muPS);
        this->alphaEW_.push_back(ev->alphaEW_);
        this->alphaS_.push_back(ev->alphaS_);
        for (auto prtcl : *ev)
        {
            this->momenta_.push_back(prtcl.momentum_);
            this->mass_.push_back(prtcl.mass_);
            this->vtim_.push_back(prtcl.vtim_);
            this->spin_.push_back(prtcl.spin_);
            this->pdg_.push_back(prtcl.pdg_);
            this->status_.push_back(prtcl.status_);
            this->mother_.push_back(prtcl.mother_);
            this->icol_.push_back(prtcl.icol_);
        }
        this->wgts_.push_back(ev->wgts_);
        this->add_extra(ev->extra);
        this->events.push_back(ev);
        return *this;
    }

    process &process::add_event_filtered(const event &ev)
    {
        auto ev_view = ev.view();
        auto summed_n = (this->n_summed.empty() ? 0 : this->n_summed.back()) + ev_view.size();
        this->n_.push_back(ev_view.size());
        this->n_summed.push_back(summed_n);
        this->proc_id_.push_back(ev.proc_id_);
        this->weight_.push_back(ev.weight_);
        this->scale_.push_back(ev.scale_);
        auto muF = (ev.muF_ == 0.0) ? ev.scale_ : ev.muF_;
        auto muR = (ev.muR_ == 0.0) ? ev.scale_ : ev.muR_;
        auto muPS = (ev.muPS_ == 0.0) ? ev.scale_ : ev.muPS_;
        this->muF_.push_back(muF);
        this->muR_.push_back(muR);
        this->muPS_.push_back(muPS);
        this->alphaEW_.push_back(ev.alphaEW_);
        this->alphaS_.push_back(ev.alphaS_);
        for (auto prtcl : ev_view)
        {
            this->momenta_.push_back(prtcl.momentum_);
            this->mass_.push_back(prtcl.mass_);
            this->vtim_.push_back(prtcl.vtim_);
            this->spin_.push_back(prtcl.spin_);
            this->pdg_.push_back(prtcl.pdg_);
            this->status_.push_back(prtcl.status_);
            this->mother_.push_back(prtcl.mother_);
            this->icol_.push_back(prtcl.icol_);
        }
        this->wgts_.push_back(ev.wgts_);
        this->add_extra(ev.extra);
        this->events.push_back(std::make_shared<event>(ev));
        return *this;
    }

    process &process::add_event_filtered(std::shared_ptr<event> ev)
    {
        auto ev_view = ev->view();
        auto summed_n = (this->n_summed.empty() ? 0 : this->n_summed.back()) + ev_view.size();
        this->n_.push_back(ev_view.size());
        this->n_summed.push_back(summed_n);
        this->proc_id_.push_back(ev->proc_id_);
        this->weight_.push_back(ev->weight_);
        this->scale_.push_back(ev->scale_);
        auto muF = (ev->muF_ == 0.0) ? ev->scale_ : ev->muF_;
        auto muR = (ev->muR_ == 0.0) ? ev->scale_ : ev->muR_;
        auto muPS = (ev->muPS_ == 0.0) ? ev->scale_ : ev->muPS_;
        this->muF_.push_back(muF);
        this->muR_.push_back(muR);
        this->muPS_.push_back(muPS);
        this->alphaEW_.push_back(ev->alphaEW_);
        this->alphaS_.push_back(ev->alphaS_);
        for (auto prtcl : ev_view)
        {
            this->momenta_.push_back(prtcl.momentum_);
            this->mass_.push_back(prtcl.mass_);
            this->vtim_.push_back(prtcl.vtim_);
            this->spin_.push_back(prtcl.spin_);
            this->pdg_.push_back(prtcl.pdg_);
            this->status_.push_back(prtcl.status_);
            this->mother_.push_back(prtcl.mother_);
            this->icol_.push_back(prtcl.icol_);
        }
        this->wgts_.push_back(ev->wgts_);
        this->add_extra(ev->extra);
        this->events.push_back(ev);
        return *this;
    }

    process &process::add_event(const event &ev)
    {
        if (this->filter)
        {
            return this->add_event_filtered(ev);
        }
        else
        {
            return this->add_event_raw(ev);
        }
    }

    process &process::add_event(std::shared_ptr<event> ev)
    {
        if (this->filter)
        {
            return this->add_event_filtered(ev);
        }
        else
        {
            return this->add_event_raw(ev);
        }
    }

    process &process::add_event(const std::vector<event> &evs)
    {
        for (const auto &ev : evs)
        {
            this->add_event(ev);
        }
        return *this;
    }

    process &process::add_event(std::vector<std::shared_ptr<event>> evs)
    {
        for (auto ev : evs)
        {
            this->add_event(ev);
        }
        return *this;
    }

    std::vector<double> process::E()
    {
        std::vector<double> energies;
        energies.reserve(this->momenta_.size());
        for (const auto &p : this->momenta_)
        {
            energies.push_back(p[0]); // p[0] is defined to be the energy, using the (1,-1,-1,-1) Minkowski metric convention
        }
        return energies;
    }
    std::vector<double> process::t() { return this->E(); }

    std::vector<double> process::x()
    {
        std::vector<double> x_;
        x_.reserve(this->momenta_.size());
        for (const auto &p : this->momenta_)
        {
            x_.push_back(p[1]); // p[1] is defined to be the x component
        }
        return x_;
    }
    std::vector<double> process::px() { return this->x(); }

    std::vector<double> process::y()
    {
        std::vector<double> y_;
        y_.reserve(this->momenta_.size());
        for (const auto &p : this->momenta_)
        {
            y_.push_back(p[2]); // p[2] is defined to be the y component
        }
        return y_;
    }
    std::vector<double> process::py() { return this->y(); }

    std::vector<double> process::z()
    {
        std::vector<double> z_;
        z_.reserve(this->momenta_.size());
        for (const auto &p : this->momenta_)
        {
            z_.push_back(p[3]); // p[3] is defined to be the z component
        }
        return z_;
    }
    std::vector<double> process::pz() { return this->z(); }

    process &process::set_E(const std::vector<double> &E)
    {
        if (E.size() != this->momenta_.size())
        {
            throw std::runtime_error("process::set_E: size mismatch");
        }
        for (size_t i = 0; i < E.size(); ++i)
        {
            this->momenta_[i][0] = E[i];
        }
        return *this;
    }
    process &process::set_t(const std::vector<double> &pt) { return this->set_E(pt); }

    process &process::set_x(const std::vector<double> &x)
    {
        if (x.size() != this->momenta_.size())
        {
            throw std::runtime_error("process::set_x: size mismatch");
        }
        for (size_t i = 0; i < x.size(); ++i)
        {
            this->momenta_[i][1] = x[i];
        }
        return *this;
    }
    process &process::set_px(const std::vector<double> &px) { return this->set_x(px); }

    process &process::set_y(const std::vector<double> &y)
    {
        if (y.size() != this->momenta_.size())
        {
            throw std::runtime_error("process::set_y: size mismatch");
        }
        for (size_t i = 0; i < y.size(); ++i)
        {
            this->momenta_[i][2] = y[i];
        }
        return *this;
    }
    process &process::set_py(const std::vector<double> &py) { return this->set_y(py); }

    process &process::set_z(const std::vector<double> &z)
    {
        if (z.size() != this->momenta_.size())
        {
            throw std::runtime_error("process::set_z: size mismatch");
        }
        for (size_t i = 0; i < z.size(); ++i)
        {
            this->momenta_[i][3] = z[i];
        }
        return *this;
    }
    process &process::set_pz(const std::vector<double> &pz) { return this->set_z(pz); }

    std::vector<double> process::gS()
    {
        std::vector<double> gS_;
        gS_.resize(this->alphaS_.size());
        std::transform(this->alphaS_.begin(), this->alphaS_.end(), gS_.begin(), [](double alpha)
                       { return std::sqrt(4 * pi * alpha); });
        return gS_;
    }

    process &process::set_gS(const std::vector<double> &gS)
    {
        if (gS.size() != this->alphaS_.size() && !this->alphaS_.empty())
        {
            warning("process::set_gS: size mismatch. process::alphaS will be overwritten and may lead to future indexing errors.");
        }
        this->alphaS_.clear();
        this->alphaS_.resize(gS.size());
        std::transform(gS.begin(), gS.end(), this->alphaS_.begin(), [](double gS)
                       { return gS * gS / (4. * pi); });
        return *this;
    }

    process &process::set_n(const std::vector<size_t> &n)
    {
        this->n_ = n;
        this->n_summed.clear();
        this->n_summed.reserve(n.size());
        size_t summed_n = 0;
        for (const auto &n_i : n)
        {
            summed_n += n_i;
            this->n_summed.push_back(summed_n);
        }
        return *this;
    }

    process &process::set_n_summed(const std::vector<size_t> &n_summed)
    {
        this->n_summed = n_summed;
        return *this;
    }

    process &process::set_proc_id(const std::vector<long int> &proc_id)
    {
        this->proc_id_ = proc_id;
        return *this;
    }

    process &process::set_weight(const std::vector<double> &weight)
    {
        this->weight_ = weight;
        return *this;
    }

    process &process::set_scale(const std::vector<double> &scale)
    {
        this->scale_ = scale;
        return *this;
    }

    process &process::set_muF(const std::vector<double> &muF)
    {
        this->muF_ = muF;
        return *this;
    }

    process &process::set_muR(const std::vector<double> &muR)
    {
        this->muR_ = muR;
        return *this;
    }

    process &process::set_muPS(const std::vector<double> &muPS)
    {
        this->muPS_ = muPS;
        return *this;
    }

    process &process::set_alphaEW(const std::vector<double> &alphaEW)
    {
        this->alphaEW_ = alphaEW;
        return *this;
    }

    process &process::set_alphaS(const std::vector<double> &alphaS)
    {
        this->alphaS_ = alphaS;
        return *this;
    }

    process &process::set_momenta(const vecArr4<double> &momenta)
    {
        this->momenta_ = momenta;
        return *this;
    }

    process &process::set_mass(const std::vector<double> &mass)
    {
        this->mass_ = mass;
        return *this;
    }

    process &process::set_vtim(const std::vector<double> &vtim)
    {
        this->vtim_ = vtim;
        return *this;
    }

    process &process::set_spin(const std::vector<double> &spin)
    {
        this->spin_ = spin;
        return *this;
    }

    process &process::set_pdg(const std::vector<long int> &pdg)
    {
        this->pdg_ = pdg;
        return *this;
    }

    process &process::set_status(const std::vector<short int> &status)
    {
        this->status_ = status;
        return *this;
    }

    process &process::set_mother(const vecArr2<short int> &mother)
    {
        this->mother_ = mother;
        return *this;
    }

    process &process::set_icol(const vecArr2<short int> &icol)
    {
        this->icol_ = icol;
        return *this;
    }

    process &process::set_wgts(const std::vector<std::vector<double>> &wgts)
    {
        this->wgts_ = wgts;
        return *this;
    }

    // Note: append_wgts() appends one weight to each existing wgt vector
    // To add an additional wgt vector, access the wgts vector directly
    // Assumes incoming weights are ordered according to the event index
    // and appends zero weights if the event index is greater than the current number of weight vectors
    process &process::append_wgts(const std::vector<double> &new_wgts)
    {
        if (this->wgts_.size() < new_wgts.size())
        {
            this->wgts_.resize(new_wgts.size());
        }
        for (size_t i = 0; i < this->wgts_.size(); ++i)
        {
            if (i < new_wgts.size())
            {
                this->wgts_[i].push_back(new_wgts[i]);
            }
            else
            {
                this->wgts_[i].push_back(0.0);
            }
        }
        return *this;
    }

    process &process::add_extra(const std::string &key, const std::any &value)
    {
        auto it = this->extra.find(key);
        if (it != this->extra.end())
        {
            it->second.push_back(value);
        }
        else
        {
            this->validate();
            if (this->n_.size() > 1)
                warning("process::add_extra: adding new extra field to non-empty process. This may lead to inconsistencies when transposing between the object oriented event format and the SoA process format.\nFor future reference, please ensure that all events contain the same extra fields.\nIn this instance, " + key + " will be added with empty std::any objects up until the current event index.");
            this->extra[key] = {};
            this->extra[key].resize(this->n_.size() - 1);
            this->extra[key].emplace_back(value);
        }
        return *this;
    }

    void process::validate() const
    {
        auto n_size = this->n_.size();
        if (n_size == 0)
        {
            if (!this->n_summed.empty() || !this->momenta_.empty() ||
                !this->mass_.empty() || !this->vtim_.empty() ||
                !this->spin_.empty() || !this->pdg_.empty() ||
                !this->proc_id_.empty() || !this->weight_.empty() ||
                !this->scale_.empty() || !this->muF_.empty() ||
                !this->muR_.empty() || !this->muPS_.empty() ||
                !this->alphaEW_.empty() || !this->alphaS_.empty())
            {
                throw std::runtime_error("process::validate() failed: n is empty but other vectors are not");
            }
        }

        auto check_event_size = [n_size](const auto &vec, const char *name)
        {
            if (vec.size() != n_size)
            {
                std::ostringstream oss;
                oss << "process::validate() failed: '" << name
                    << "' has size " << vec.size() << ", expected " << n_size;
                throw std::runtime_error(oss.str());
            }
        };

        auto check_mu_size = [n_size](const auto &vec, const char *name)
        {
            if (vec.size() != n_size && !vec.empty())
            {
                std::ostringstream oss;
                oss << "process::validate() failed: '" << name
                    << "' has size " << vec.size() << ", expected " << n_size;
                throw std::runtime_error(oss.str());
            }
        };

        check_event_size(this->n_, "n");
        check_event_size(this->n_summed, "n_summed");
        check_event_size(this->proc_id_, "proc_id");
        check_event_size(this->weight_, "weight");
        check_event_size(this->scale_, "scale");
        check_mu_size(this->muF_, "muF");
        check_mu_size(this->muR_, "muR");
        check_mu_size(this->muPS_, "muPS");
        check_event_size(this->alphaEW_, "alphaEW");
        check_event_size(this->alphaS_, "alphaS");
        check_event_size(this->wgts_, "wgts");

        size_t summed_n = this->n_summed.empty() ? 0 : this->n_summed.back();

        size_t explicit_n_summed = std::accumulate(this->n_.begin(), this->n_.end(), 0);

        if (summed_n != explicit_n_summed)
        {
            throw std::runtime_error("process::validate() failed: n_summed does not match sum of n");
        }

        auto check_particle_size = [summed_n](const auto &vec, const char *name)
        {
            if (vec.size() != summed_n)
            {
                std::ostringstream oss;
                oss << "process::validate() failed: '" << name
                    << "' has size " << vec.size() << ", expected " << summed_n;
                throw std::runtime_error(oss.str());
            }
        };

        check_particle_size(this->momenta_, "momenta");
        check_particle_size(this->mass_, "mass");
        check_particle_size(this->vtim_, "vtim");
        check_particle_size(this->spin_, "spin");
        check_particle_size(this->pdg_, "pdg");
        check_particle_size(this->status_, "status");
        check_particle_size(this->mother_, "mother");
        check_particle_size(this->icol_, "icol");
    }

    process &process::add_extra(const std::unordered_map<std::string, std::any> &extras)
    {
        for (const auto &[key, value] : extras)
        {
            this->add_extra(key, value);
        }
        return *this;
    }

    void process::make_event(size_t idx)
    {
        if (idx >= this->n_.size())
        {
            throw std::out_of_range("Invalid event index");
        }
        std::shared_ptr<event> ev = std::make_shared<event>();
        ev->set_n(this->n_[idx]).set_proc_id(this->proc_id_[idx]).set_weight(this->weight_[idx]).set_scale(this->scale_[idx]).set_alphaEW(this->alphaEW_[idx]).set_alphaS(this->alphaS_[idx]).set_wgts(this->wgts_[idx]);
        if (this->muF_.size() > idx)
            if (this->muF_[idx] > 1e-08 && std::abs(this->muF_[idx] - this->scale_[idx]) > 1e-08)
            {
                ev->set_muF(this->muF_[idx]);
            }
        if (this->muR_.size() > idx)
            if (this->muR_[idx] > 1e-08 && std::abs(this->muR_[idx] - this->scale_[idx]) > 1e-08)
            {
                ev->set_muR(this->muR_[idx]);
            }
        if (this->muPS_.size() > idx)
            if (this->muPS_[idx] > 1e-08 && std::abs(this->muPS_[idx] - this->scale_[idx]) > 1e-08)
            {
                ev->set_muPS(this->muPS_[idx]);
            }
        size_t begin;
        if (idx == 0)
        {
            begin = 0;
        }
        else
        {
            begin = this->n_summed[idx - 1];
        }
        size_t end = this->n_summed[idx];
        ev->set_momenta(this->momenta_.subvec(begin, end)).set_mass(subvector(this->mass_, begin, end)).set_vtim(subvector(this->vtim_, begin, end)).set_spin(subvector(this->spin_, begin, end)).set_pdg(subvector(this->pdg_, begin, end)).set_status(subvector(this->status_, begin, end)).set_mother(this->mother_.subvec(begin, end)).set_icol(this->icol_.subvec(begin, end));
        for (auto &[key, values] : this->extra)
        {
            if (values.size() < this->n_.size())
            {
                values.resize(this->n_.size());
            }
            ev->set(key, values[idx]);
        }
        if (idx >= this->events.size())
        {
            events.resize(idx + 1);
            events[idx] = ev;
        }
        else
        {
            events[idx] = ev;
        }
    }

    process &process::add_extra(const std::string &key, const std::vector<std::any> &values)
    {
        if (values.size() < this->n_.size())
        {
            warning("process::add_extra() - Resizing vector for key: " + key + " to match number of events indicated by n.size().\nFor future reference, please ensure that data sizes match to avoid inconsistencies in transpositions.");
        }
        this->extra[key] = values;
        this->extra[key].resize(this->n_.size());
        return *this;
    }

    process &process::add_extra(const std::unordered_map<std::string, std::vector<std::any>> &values)
    {
        for (const auto &[key, val] : values)
        {
            this->extra[key] = val;
            if (val.size() < this->n_.size())
            {
                warning("process::add_extra() - Resizing vector for key: " + key + " to match number of events indicated by n.size().\nFor future reference, please ensure that data sizes match to avoid inconsistencies in transpositions.");
                this->extra[key].resize(this->n_.size());
            }
        }
        return *this;
    }

    process &process::set_extra(const std::unordered_map<std::string, std::vector<std::any>> &values)
    {
        this->extra.clear();
        this->extra.reserve(values.size());
        for (const auto &[key, val] : values)
        {
            this->extra[key] = val;
            if (val.size() < this->n_.size())
            {
                warning("process::set_extra() - Resizing vector for key: " + key + " to match number of events indicated by n.size().\nFor future reference, please ensure that data sizes match to avoid inconsistencies in transpositions.");
                this->extra[key].resize(this->n_.size());
            }
        }
        return *this;
    }

    process &process::set_filter(bool v)
    {
        this->filter = v;
        return *this;
    }

    // Function to transpose process information into the OOP event format
    void process::transpose()
    {
        this->validate();
        this->events.clear();
        for (size_t i = 0; i < this->n_.size(); ++i)
        {
            this->make_event(i);
        }
    }

    // Functions for partial transpositions
    process &process::transpose_n()
    {
        this->validate();
        if (this->n_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_n() - Number of events does not match number of n values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            this->events[i]->set_n(this->n_[i]);
        }
        return *this;
    }
    process &process::transpose_nUP() { return this->transpose_n(); }

    process &process::transpose_proc_id()
    {
        this->validate();
        if (this->proc_id_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_proc_id() - Number of events does not match number of proc_id values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            this->events[i]->set_proc_id(this->proc_id_[i]);
        }
        return *this;
    }
    process &process::transpose_idPrUP() { return this->transpose_proc_id(); }
    process &process::transpose_idPr() { return this->transpose_proc_id(); }

    process &process::transpose_weight()
    {
        this->validate();
        if (this->weight_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_weight() - Number of events does not match number of weight values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            this->events[i]->set_weight(this->weight_[i]);
        }
        return *this;
    }
    process &process::transpose_xWgtUP() { return this->transpose_weight(); }
    process &process::transpose_xWgt() { return this->transpose_weight(); }

    process &process::transpose_scale()
    {
        this->validate();
        if (this->scale_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_scale() - Number of events does not match number of scale values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            this->events[i]->set_scale(this->scale_[i]);
        }
        return *this;
    }
    process &process::transpose_scalUP() { return this->transpose_scale(); }

    process &process::transpose_muF()
    {
        this->validate();
        if (this->muF_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_muF() - Number of events does not match number of muF values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            if (this->muF_[i] > 1e-08 && std::abs(this->muF_[i] - this->scale_[i]) > 1e-08)
            {
                this->events[i]->set_muF(this->muF_[i]);
            }
            else
            {
                this->events[i]->set_muF(0.0);
            }
        }
        return *this;
    }

    process &process::transpose_muR()
    {
        this->validate();
        if (this->muR_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_muR() - Number of events does not match number of muR values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            if (this->muR_[i] > 1e-08 && std::abs(this->muR_[i] - this->scale_[i]) > 1e-08)
            {
                this->events[i]->set_muR(this->muR_[i]);
            }
            else
            {
                this->events[i]->set_muR(0.0);
            }
        }
        return *this;
    }

    process &process::transpose_muPS()
    {
        this->validate();
        if (this->muPS_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_muPS() - Number of events does not match number of muPS values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            if (this->muPS_[i] > 1e-08 && std::abs(this->muPS_[i] - this->scale_[i]) > 1e-08)
            {
                this->events[i]->set_muPS(this->muPS_[i]);
            }
            else
            {
                this->events[i]->set_muPS(0.0);
            }
        }
        return *this;
    }

    process &process::transpose_alphaEW()
    {
        this->validate();
        if (this->alphaEW_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_alphaEW() - Number of events does not match number of alphaEW values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            if (this->alphaEW_[i] > 1e-08 && std::abs(this->alphaEW_[i] - this->scale_[i]) > 1e-08)
            {
                this->events[i]->set_alphaEW(this->alphaEW_[i]);
            }
            else
            {
                this->events[i]->set_alphaEW(0.0);
            }
        }
        return *this;
    }
    process &process::transpose_aQEDUP() { return this->transpose_alphaEW(); }
    process &process::transpose_aQED() { return this->transpose_alphaEW(); }

    process &process::transpose_alphaS()
    {
        this->validate();
        if (this->alphaS_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_alphaS() - Number of events does not match number of alphaS values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            if (this->alphaS_[i] > 1e-08 && std::abs(this->alphaS_[i] - this->scale_[i]) > 1e-08)
            {
                this->events[i]->set_alphaS(this->alphaS_[i]);
            }
            else
            {
                this->events[i]->set_alphaS(0.0);
            }
        }
        return *this;
    }
    process &process::transpose_aQCDUP() { return this->transpose_alphaS(); }
    process &process::transpose_aQCD() { return this->transpose_alphaS(); }

    process &process::transpose_momenta()
    {
        this->validate();
        if (this->momenta_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_momenta() - Number of events does not match number of momenta values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            this->events[i]->set_momenta(this->momenta_.subvec(begin, end));
        }
        return *this;
    }
    process &process::transpose_pUP() { return this->transpose_momenta(); }
    process &process::transpose_mom() { return this->transpose_momenta(); }
    process &process::transpose_p() { return this->transpose_momenta(); }

    process &process::transpose_mass()
    {
        this->validate();
        if (this->mass_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_mass() - Number of events does not match number of mass values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            this->events[i]->set_mass(subvector(this->mass_, begin, end));
        }
        return *this;
    }
    process &process::transpose_mUP() { return this->transpose_mass(); }
    process &process::transpose_m() { return this->transpose_mass(); }

    process &process::transpose_vtim()
    {
        this->validate();
        if (this->vtim_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_vtim() - Number of events does not match number of vtim values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            this->events[i]->set_vtim(subvector(this->vtim_, begin, end));
        }
        return *this;
    }
    process &process::transpose_vTimUP() { return this->transpose_vtim(); }

    process &process::transpose_spin()
    {
        this->validate();
        if (this->spin_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_spin() - Number of events does not match number of spin values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->n_summed.back(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            this->events[i]->set_spin(subvector(this->spin_, begin, end));
        }
        return *this;
    }
    process &process::transpose_spinUP() { return this->transpose_spin(); }

    process &process::transpose_pdg()
    {
        this->validate();
        if (this->pdg_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_pdg() - Number of events does not match number of pdg values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            this->events[i]->set_pdg(subvector(this->pdg_, begin, end));
        }
        return *this;
    }
    process &process::transpose_idUP() { return this->transpose_pdg(); }
    process &process::transpose_id() { return this->transpose_pdg(); }

    process &process::transpose_status()
    {
        this->validate();
        if (this->status_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_status() - Number of events does not match number of status values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            this->events[i]->set_status(subvector(this->status_, begin, end));
        }
        return *this;
    }
    process &process::transpose_iStUP() { return this->transpose_status(); }
    process &process::transpose_iSt() { return this->transpose_status(); }

    process &process::transpose_mother()
    {
        this->validate();
        if (this->mother_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_mother() - Number of events does not match number of mother values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            this->events[i]->set_mother(this->mother_.subvec(begin, end));
        }
        return *this;
    }
    process &process::transpose_mothUP() { return this->transpose_mother(); }

    process &process::transpose_icol()
    {
        this->validate();
        if (this->icol_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_icol() - Number of events does not match number of icol values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            this->events[i]->set_icol(this->icol_.subvec(begin, end));
        }
        return *this;
    }
    process &process::transpose_iColUP() { return this->transpose_icol(); }

    process &process::transpose_wgts()
    {
        this->validate();
        if (this->wgts_.size() != this->events.size())
        {
            if (!this->events.empty())
                warning("process::transpose_wgts() - Number of events does not match number of wgts values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            this->events[i]->set_wgts(this->wgts_[i]);
        }
        return *this;
    }

    process &process::transpose_extra()
    {
        this->validate();
        if (this->extra.empty())
        {
            warning("process::transpose_extra() - No extra fields to transpose.");
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            for (const auto &[key, values] : this->extra)
            {
                if (values.size() <= i)
                {
                    warning("process::transpose_extra() - Not enough values for key: " + key + ". Resizing to match number of events.");
                    this->extra[key].resize(this->events.size());
                }
                this->events[i]->set(key, values[i]);
            }
        }
        return *this;
    }

    process &process::transpose_E()
    {
        this->validate();
        if (this->momenta_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_E() - Number of events does not match number of E values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            auto full_mom = this->momenta_.subvec(begin, end);
            if (full_mom.size() != this->events.size())
            {
                throw std::runtime_error("process::transpose_E() - Size mismatch between momenta and events.");
            }
            for (size_t j = 0; j < full_mom.size(); ++j)
            {
                this->events[i]->at(j).E() = full_mom[j][0];
            }
        }
        return *this;
    }
    process &process::transpose_t() { return this->transpose_E(); }

    process &process::transpose_x()
    {
        this->validate();
        if (this->momenta_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_x() - Number of events does not match number of x values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            auto full_mom = this->momenta_.subvec(begin, end);
            if (full_mom.size() != this->events.size())
            {
                throw std::runtime_error("process::transpose_x() - Size mismatch between momenta and events.");
            }
            for (size_t j = 0; j < full_mom.size(); ++j)
            {
                this->events[i]->at(j).px() = full_mom[j][1];
            }
        }
        return *this;
    }
    process &process::transpose_px() { return this->transpose_x(); }

    process &process::transpose_y()
    {
        this->validate();
        if (this->momenta_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_y() - Number of events does not match number of y values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            auto full_mom = this->momenta_.subvec(begin, end);
            if (full_mom.size() != this->events.size())
            {
                throw std::runtime_error("process::transpose_y() - Size mismatch between momenta and events.");
            }
            for (size_t j = 0; j < full_mom.size(); ++j)
            {
                this->events[i]->at(j).py() = full_mom[j][2];
            }
        }
        return *this;
    }
    process &process::transpose_py() { return this->transpose_y(); }

    process &process::transpose_z()
    {
        this->validate();
        if (this->momenta_.size() != this->n_summed.back())
        {
            if (!this->events.empty())
                warning("process::transpose_z() - Number of events does not match number of z values. Overwriting process::events vector.");
            this->transpose();
            return *this;
        }
        for (size_t i = 0; i < this->events.size(); ++i)
        {
            size_t begin;
            if (i == 0)
            {
                begin = 0;
            }
            else
            {
                begin = this->n_summed[i - 1];
            }
            size_t end = this->n_summed[i];
            auto full_mom = this->momenta_.subvec(begin, end);
            if (full_mom.size() != this->n_summed.back())
            {
                throw std::runtime_error("process::transpose_z() - Size mismatch between momenta and events.");
            }
            for (size_t j = 0; j < full_mom.size(); ++j)
            {
                this->events[i]->at(j).pz() = full_mom[j][3];
            }
        }
        return *this;
    }
    process &process::transpose_pz() { return this->transpose_z(); }

    initNode::initNode(short unsigned int nproc)
    {
        this->nProc_ = nproc;
        this->xSec_.resize(nproc, 0.0);
        this->xSecErr_.resize(nproc, 0.0);
        this->xMax_.resize(nproc, 0.0);
        this->lProc_.resize(nproc, 0);
    }

    initNode::initNode(size_t nproc)
    {
        // Check that nproc fits in a short
        if (nproc > std::numeric_limits<short unsigned int>::max())
        {
            throw std::invalid_argument("initNode::initNode() - nproc is too large");
        }
        this->nProc_ = static_cast<short unsigned int>(nproc);
        this->xSec_.resize(nproc, 0.0);
        this->xSecErr_.resize(nproc, 0.0);
        this->xMax_.resize(nproc, 0.0);
        this->lProc_.resize(nproc, 0);
    }

    void initNode::validate_init() const
    {
        if (this->nProc_ != this->xSec_.size())
        {
            throw std::runtime_error("initNode::validate_init() failed: nProc does not match size of xSec");
        }
        if (this->nProc_ != this->xSecErr_.size())
        {
            throw std::runtime_error("initNode::validate_init() failed: nProc does not match size of xSecErr");
        }
        if (this->nProc_ != this->xMax_.size())
        {
            throw std::runtime_error("initNode::validate_init() failed: nProc does not match size of xMax");
        }
        if (this->nProc_ != this->lProc_.size())
        {
            throw std::runtime_error("initNode::validate_init() failed: nProc does not match size of lProc");
        }
    }

    void initNode::print_head(std::ostream &os) const
    {
        os << std::scientific << std::setprecision(6)
           << this->idBm_[0] << " " << this->idBm_[1]
           << " " << this->eBm_[0] << " " << this->eBm_[1]
           << " " << this->pdfG_[0] << " " << this->pdfG_[1]
           << " " << this->pdfS_[0] << " " << this->pdfS_[1]
           << " " << this->idWgt_
           << " " << this->nProc_
           << "\n";
    }

    void initNode::print_body(std::ostream &os) const
    {
        os << std::scientific << std::setprecision(6);
        for (size_t i = 0; i < this->xSec_.size(); ++i)
        {
            os << this->xSec_[i] << " " << this->xSecErr_[i]
               << " " << this->xMax_[i] << " " << this->lProc_[i]
               << "\n";
        }
    }

    void initNode::print_extra(std::ostream &os) const
    {
        for (const auto &[key, val] : this->extra)
        {
            if (val.type() == typeid(int))
            {
                os << "<" << key << "\">" << std::any_cast<int>(val) << "</" << key << ">\n";
            }
            else if (val.type() == typeid(double))
            {
                os << std::setprecision(10) << std::scientific;
                os << "<" << key << "\">" << std::any_cast<double>(val) << "</" << key << ">\n";
            }
            else if (val.type() == typeid(std::string))
            {
                os << "<" << key << "\">" << std::any_cast<std::string>(val) << "</" << key << ">\n";
            }
            else if (val.type() == typeid(std::string_view))
            {
                os << "<" << key << "\">" << std::any_cast<std::string_view>(val) << "</" << key << ">\n";
            }
            else if (val.type() == typeid(xmlNode))
            {
                std::any_cast<xmlNode>(val).write(os);
                os << "\n";
            }
            else if (val.type() == typeid(std::shared_ptr<xmlNode>))
            {
                std::any_cast<std::shared_ptr<xmlNode>>(val)->write(os);
                os << "\n";
            }
            else
            {
                warning("Unknown type for extra field: " + key + ", skipping print");
            }
        }
    }

    void initNode::print_init(std::ostream &os) const
    {
        this->validate_init();
        os << "\n<init>\n";
        this->print_head(os);
        this->print_body(os);
        this->print_extra(os);
        os << "</init>";
    }

    lhe::lhe(std::vector<std::shared_ptr<event>> evs)
    {
        this->events = evs;
        std::vector<long int> proc_ids = {};
        this->sync_weight_ids();
        for (auto ev : evs)
        {
            if (std::find(proc_ids.begin(), proc_ids.end(), ev->proc_id_) == proc_ids.end())
            {
                proc_ids.push_back(ev->proc_id_);
            }
        }
        this->nProc_ = proc_ids.size();
        std::sort(proc_ids.begin(), proc_ids.end());
        this->lProc_ = proc_ids;
    }

    lhe::lhe(std::vector<event> evs)
    {
        this->events.reserve(evs.size());
        for (const auto &ev : evs)
        {
            this->events.push_back(std::make_shared<event>(ev));
        }
        this->sync_weight_ids();
        std::vector<long int> proc_ids = {};
        for (auto ev : this->events)
        {
            if (std::find(proc_ids.begin(), proc_ids.end(), ev->proc_id_) == proc_ids.end())
            {
                proc_ids.push_back(ev->proc_id_);
            }
        }
        this->nProc_ = proc_ids.size();
        std::sort(proc_ids.begin(), proc_ids.end());
        this->lProc_ = proc_ids;
    }

    lhe::lhe(const initNode &i, std::vector<std::shared_ptr<event>> evts) : initNode(i), events(std::move(evts))
    {
        std::vector<long int> proc_ids = {};
        this->sync_weight_ids();
        for (auto ev : this->events)
        {
            if (std::find(proc_ids.begin(), proc_ids.end(), ev->proc_id_) == proc_ids.end())
            {
                proc_ids.push_back(ev->proc_id_);
            }
        }
        this->nProc_ = proc_ids.size();
        std::sort(proc_ids.begin(), proc_ids.end());
        this->lProc_ = proc_ids;
    }

    lhe::lhe(const initNode &i, std::vector<event> evts) : initNode(i)
    {
        this->events.reserve(evts.size());
        for (const auto &ev : evts)
        {
            this->events.push_back(std::make_shared<event>(ev));
        }
        this->sync_weight_ids();
        std::vector<long int> proc_ids = {};
        for (auto ev : this->events)
        {
            if (std::find(proc_ids.begin(), proc_ids.end(), ev->proc_id_) == proc_ids.end())
            {
                proc_ids.push_back(ev->proc_id_);
            }
        }
        this->nProc_ = proc_ids.size();
        std::sort(proc_ids.begin(), proc_ids.end());
        this->lProc_ = proc_ids;
    }

    lhe &lhe::set_events(std::vector<std::shared_ptr<event>> evs)
    {
        this->events = evs;
        this->sync_weight_ids();
        return *this;
    }

    lhe &lhe::set_processes(std::vector<std::shared_ptr<process>> procs)
    {
        this->processes = procs;
        return *this;
    }

    lhe &lhe::set_header(std::any hdr)
    {
        this->header = std::move(hdr);
        return *this;
    }

    lhe &lhe::set_weight_ids(const std::vector<std::string> &ids)
    {
        *this->weight_ids = ids;
        return *this;
    }

    lhe &lhe::set_weight_ids(std::vector<std::string> &&ids)
    {
        *this->weight_ids = std::move(ids);
        return *this;
    }

    lhe &lhe::set_weight_ids(std::shared_ptr<std::vector<std::string>> ids)
    {
        this->weight_ids = std::move(ids);
        return *this;
    }

    lhe &lhe::add_weight_id(const std::string &id)
    {
        this->weight_ids->push_back(id);
        return *this;
    }

    lhe &lhe::add_weight_id(std::string &&id)
    {
        this->weight_ids->push_back(std::move(id));
        return *this;
    }

    void lhe::extract_weight_ids()
    {
        if (!this->weight_ids)
        {
            this->weight_ids = std::make_shared<std::vector<std::string>>();
        }
        if (this->header.type() != typeid(std::shared_ptr<xmlNode>))
        {
            throw std::runtime_error("lhe::extract_weight_ids() - Header is not of type std::shared_ptr<xmlNode>");
        }
        auto xml_header = std::any_cast<std::shared_ptr<xmlNode>>(this->header);
        auto id_puller = [&](std::shared_ptr<xmlNode> node)
        {
            auto atrs = node->attrs();
            for (auto atr : atrs)
            {
                if (atr.name() == "id")
                    this->weight_ids->push_back(std::string(atr.value()));
            }
        };
        // Extract weight IDs from the XML header
        auto initrwgt = xml_header->get_child("initrwgt");
        if (initrwgt)
        {
            for (auto child : initrwgt->children())
            {
                if (child->name() == "weight")
                {
                    id_puller(child);
                }
                else if (child->name() == "weightgroup")
                {
                    for (auto grandchild : child->children())
                    {
                        if (grandchild->name() == "weight")
                        {
                            id_puller(grandchild);
                        }
                    }
                }
            }
        }
        for (auto ev : this->events)
        {
            ev->weight_ids = this->weight_ids; // overwrites weight_ids if original set has several different ones
        }
    }

    void lhe::sync_weight_ids()
    {
        if (!this->weight_ids)
        {
            this->weight_ids = std::make_shared<std::vector<std::string>>();
        }
        for (auto ev : this->events)
        {
            if (!ev)
                continue;
            if (ev->weight_ids)
            { // If event has weight_ids, check if it's larger than current
                if (ev->weight_ids->size() > this->weight_ids->size())
                    this->weight_ids = ev->weight_ids;
            }
        }
        for (auto ev : this->events)
        {
            ev->weight_ids = this->weight_ids; // overwrites weight_ids if original set has several different ones
        }
    }

    initNode &initNode::set_idBm(const arr2<long int> &idBm)
    {
        this->idBm_ = idBm;
        return *this;
    }

    initNode &initNode::set_idBm(long int id1, long int id2)
    {
        this->idBm_ = {id1, id2};
        return *this;
    }

    initNode &initNode::set_eBm(const arr2<double> &energies)
    {
        this->eBm_ = energies;
        return *this;
    }

    initNode &initNode::set_eBm(double e1, double e2)
    {
        this->eBm_ = {e1, e2};
        return *this;
    }

    initNode &initNode::set_pdfG(const arr2<short int> &pdfG)
    {
        this->pdfG_ = pdfG;
        return *this;
    }

    initNode &initNode::set_pdfG(short int pdf1, short int pdf2)
    {
        this->pdfG_ = {pdf1, pdf2};
        return *this;
    }

    initNode &initNode::set_pdfS(const arr2<long int> &pdfS)
    {
        this->pdfS_ = pdfS;
        return *this;
    }

    initNode &initNode::set_pdfS(long int pdf1, long int pdf2)
    {
        this->pdfS_ = {pdf1, pdf2};
        return *this;
    }

    initNode &initNode::set_idWgt(short int id)
    {
        this->idWgt_ = id;
        return *this;
    }

    initNode &initNode::set_nProc(short unsigned int n)
    {
        this->nProc_ = n;
        return *this;
    }

    initNode &initNode::set_xSec(const std::vector<double> &xSec)
    {
        this->xSec_ = xSec;
        return *this;
    }

    initNode &initNode::set_xSecErr(const std::vector<double> &xSecErr)
    {
        this->xSecErr_ = xSecErr;
        return *this;
    }
    initNode &initNode::set_xMax(const std::vector<double> &xMax)
    {
        this->xMax_ = xMax;
        return *this;
    }

    initNode &initNode::set_lProc(const std::vector<long int> &lProc)
    {
        this->lProc_ = lProc;
        return *this;
    }

    initNode &initNode::add_xSec(double xsec)
    {
        this->xSec_.push_back(xsec);
        return *this;
    }

    initNode &initNode::add_xSecErr(double xsec_err)
    {
        this->xSecErr_.push_back(xsec_err);
        return *this;
    }

    initNode &initNode::add_xMax(double xmax)
    {
        this->xMax_.push_back(xmax);
        return *this;
    }

    initNode &initNode::add_lProc(long int lproc)
    {
        this->lProc_.push_back(lproc);
        return *this;
    }

    lhe &lhe::add_event(std::shared_ptr<event> ev)
    {
        this->events.push_back(ev);
        return *this;
    }

    lhe &lhe::add_event(const event &ev)
    {
        this->events.push_back(std::make_shared<event>(ev));
        return *this;
    }

    // If no sorter is provided, sort by external partons
    lhe &lhe::set_sorter()
    {
        if (this->events.empty())
        {
            throw std::runtime_error("lhe::set_sorter() called with no events");
        }
        this->sorter = make_sample_sorter(this->events);
        this->event_hash = this->sorter.get_hash();
        return *this;
    }

    lhe &lhe::set_sorter(event_equal_fn comp)
    {
        if (this->events.empty())
        {
            throw std::runtime_error("lhe::set_sorter() called with no events");
        }
        this->sorter = make_sample_sorter(this->events, comp);
        this->event_hash = this->sorter.get_hash();
        return *this;
    }

    lhe &lhe::set_sorter(cevent_equal_fn comp)
    {
        if (this->events.empty())
        {
            throw std::runtime_error("lhe::set_sorter() called with no events");
        }
        this->sorter = make_sample_sorter(this->events, comp);
        this->event_hash = this->sorter.get_hash();
        return *this;
    }

    lhe &lhe::set_sorter(const eventSorter &sort)
    {
        this->sorter = sort;
        this->event_hash = this->sorter.get_hash();
        return *this;
    }

    void lhe::extract_hash()
    {
        if (!this->sorter.size())
            this->set_sorter();
        this->event_hash = this->sorter.get_hash();
    }

    lhe &lhe::set_hash(event_hash_fn hash)
    {
        this->event_hash = hash;
        return *this;
    }

    lhe &lhe::set_filter(bool filter)
    {
        this->filter_processes = filter;
        return *this;
    }

    void lhe::sort_events()
    {
        if (this->events.empty())
        {
            throw std::runtime_error("lhe::sort_events() called with no events");
        }
        if (this->sorter.size() == 0 && !this->event_hash)
        {
            this->set_sorter();
        }
        this->process_order.clear();
        this->process_order.reserve(this->events.size());
        for (auto ev : this->events)
        {
            this->process_order.push_back(this->event_hash(*ev));
        }
        if (this->process_order.size() != this->events.size())
        {
            throw std::runtime_error("lhe::sort_events() failed: process_order size does not match events size");
        }
        this->sorted_events.clear();
        for (size_t j = 0; j < this->sorter.size(); ++j)
        {
            this->sorted_events.push_back({});
        }
        this->sorted_events.push_back({});
        for (size_t ind = 0; ind < this->events.size(); ++ind)
        {
            size_t sort_ind = this->process_order[ind];
            if (sort_ind == npos)
                sort_ind = this->sorted_events.size() - 1; // If an event does not belong to any set in the sorter, returns npos; compensate by adding an additional "unsorted" vector at the end
            this->sorted_events[sort_ind].push_back(this->events[ind]);
        }
        // If all events were successfully sorted, remove the empty "unsorted" vector
        if (this->sorted_events.back().empty())
        {
            this->sorted_events.pop_back(); // Remove empty last set
        }
    }

    void lhe::unsort_events()
    {
        if (this->sorted_events.empty())
        {
            throw std::runtime_error("lhe::unsort_events() called with no sorted events");
        }
        this->events.clear();
        this->events.reserve(this->process_order.size());
        std::vector<size_t> unorder;
        unorder.resize(this->sorted_events.size());
        for (size_t j = 0; j < this->process_order.size(); ++j)
        {
            auto curr_idx = this->process_order[j];
            if (curr_idx == npos)
                curr_idx = unorder.size() - 1;
            auto srt_idx = unorder[curr_idx];
            if (srt_idx >= this->sorted_events[curr_idx].size())
                throw std::runtime_error("lhe::unsort_events() failed: sorted event index out of bounds");
            this->events.push_back(this->sorted_events[curr_idx][srt_idx]);
            unorder[curr_idx]++;
        }
    }

    void lhe::events_to_processes()
    {
        if (this->events.empty())
        {
            throw std::runtime_error("lhe::events_to_processes() called with no events");
        }
        this->sort_events(); // Overrides previous sorting (if one exists)
        this->processes.clear();
        for (auto ev_set : this->sorted_events)
        {
            this->processes.push_back(std::make_shared<process>(ev_set, this->filter_processes));
        }
    }

    void lhe::processes_to_events()
    {
        if (this->processes.empty())
        {
            throw std::runtime_error("lhe::processes_to_events() called with no processes");
        }
        this->sorted_events.clear();
        for (auto &proc : this->processes)
        {
            proc->transpose();
            this->sorted_events.push_back(proc->events);
        }
        this->unsort_events();
    }

    void lhe::transpose()
    {
        if (this->events.empty() && this->processes.empty())
        {
            throw std::runtime_error("lhe::transpose() called with no events or processes");
        }
        if (!this->events.empty())
        {
            // Transpose events to processes
            this->events_to_processes();
        }
        else if (!this->processes.empty())
        {
            // Transpose processes to events
            this->processes_to_events();
        }
        else
        {
            warning("lhe::transpose() called with both events and processes filled, cannot deduce which transposition to run.\nCall with string argument \"events\" to transpose from events to processes or with string argument \"processes\" to transpose from processes to events.\nAlternatively, call the lhe::events_to_processes() or lhe::processes_to_events() functions directly.");
        }
    }

    void lhe::transpose(std::string dir)
    {
        if (dir == "events" || dir == "event")
        {
            this->events_to_processes();
        }
        else if (dir == "processes" || dir == "process")
        {
            this->processes_to_events();
        }
        else
        {
            warning("lhe::transpose() called with invalid direction: " + dir + "\nCall with string argument \"events\" to transpose from events to processes or with string argument \"processes\" to transpose from processes to events.");
        }
    }

    void lhe::append_weight_ids(bool include)
    {
        if (this->weight_ids->empty())
        {
            return;
        }
        if (this->header.type() != typeid(REX::xmlNode) && this->header.type() != typeid(std::shared_ptr<REX::xmlNode>))
        {
            return;
        }
        std::string wgtgr_str = "\n<weightgroup name=\'tearex_reweighting\'";
        if (include)
            wgtgr_str += " weight_name_strategy=\'includeIdInWeightName\'>\n";
        for (size_t i = 0; i < this->weight_ids->size(); i++)
        {
            wgtgr_str += "<weight id=\'" + this->weight_ids->at(i) + "\'>";
            if (i < this->weight_context.size())
            {
                wgtgr_str += this->weight_context[i];
            }
            wgtgr_str += "</weight>\n";
        }
        wgtgr_str += "</weightgroup>\n";
        auto wgt_node = xmlNode::parse(wgtgr_str);
        std::shared_ptr<xmlNode> initrwgt;
        if (this->header.type() == typeid(REX::xmlNode))
        {
            if (std::any_cast<REX::xmlNode>(this->header).get_child("initrwgt"))
            {
                initrwgt = std::any_cast<REX::xmlNode>(this->header).get_child("initrwgt");
            }
            else
            {
                initrwgt = xmlNode::parse("<initrwgt>\n</initrwgt>\n");
            }
        }
        else if (this->header.type() == typeid(std::shared_ptr<REX::xmlNode>))
        {
            if (std::any_cast<std::shared_ptr<REX::xmlNode>>(this->header)->get_child("initrwgt"))
            {
                initrwgt = std::any_cast<std::shared_ptr<REX::xmlNode>>(this->header)->get_child("initrwgt");
            }
            else
            {
                initrwgt = xmlNode::parse("<initrwgt>\n</initrwgt>\n");
            }
        }
        else
        {
            initrwgt = xmlNode::parse("<initrwgt>\n</initrwgt>\n");
        }
        initrwgt->add_child(wgt_node, true);
        if (this->header.type() == typeid(REX::xmlNode))
        {
            this->header = std::any_cast<REX::xmlNode>(this->header).replace_child("initrwgt", initrwgt, true);
        }
        else if (this->header.type() == typeid(std::shared_ptr<REX::xmlNode>))
        {
            std::any_cast<std::shared_ptr<REX::xmlNode>>(this->header)->replace_child("initrwgt", initrwgt, true);
        }
    }

    void lhe::print_header(std::ostream &os) const
    {
        if (this->header.type() == typeid(std::string))
        {
            os << std::any_cast<std::string>(this->header);
        }
        else if (this->header.type() == typeid(std::shared_ptr<xmlNode>))
        {
            std::any_cast<std::shared_ptr<xmlNode>>(this->header)->write(os);
        }
        else if (this->header.type() == typeid(xmlNode))
        {
            std::any_cast<xmlNode>(this->header).write(os);
        }
        else if (this->header.has_value())
        {
            warning("lhe::print_header() - Header is of unknown type, cannot print.");
        }
    }

    void lhe::print(std::ostream &os, bool include_ids)
    {
        this->append_weight_ids(include_ids);
        os << "<LesHouchesEvents version=\"3.0\">\n";
        this->print_header(os);
        this->print_init(os);
        for (auto event : this->events)
        {
            event->print(os, include_ids);
        }
        os << "\n</LesHouchesEvents>";
    }

    // -------------------- xmlDoc --------------------

    xmlDoc::xmlDoc(std::string xml)
        : buf_(std::make_shared<std::string>(std::move(xml))) {}

    const std::string &xmlDoc::str() const noexcept { return *buf_; }
    std::string_view xmlDoc::view() const noexcept { return std::string_view(*buf_); }
    std::shared_ptr<const std::string> xmlDoc::shared() const noexcept { return buf_; }

    // -------------------- Attr --------------------

    std::string_view Attr::name() const noexcept { return name_new ? std::string_view(*name_new) : name_view; }
    std::string_view Attr::value() const noexcept { return value_new ? std::string_view(*value_new) : value_view; }
    bool Attr::modified() const noexcept { return name_new.has_value() || value_new.has_value(); }

    // -------------------- xmlNode (public) --------------------

    xmlNode::xmlNode() = default;
    xmlNode::~xmlNode() = default;

    // xmlNode::parse(std::string)
    std::shared_ptr<xmlNode> xmlNode::parse(std::string xml)
    {
        xmlDoc doc{std::move(xml)};
        auto shared = doc.shared();

        // Keep the original first '<' offset
        size_t first_start = find_first_element_start(*shared, 0);
        size_t pos = first_start; // pass a copy; parse_element will advance this

        auto root = parse_element(shared, pos);
        if (root)
        {
            root->prolog_start_ = 0;
            root->prolog_end_ = first_start; // **not** the advanced pos
        }
        return root;
    }

    // xmlNode::parse(const std::shared_ptr<const std::string>&)
    std::shared_ptr<xmlNode> xmlNode::parse(const std::shared_ptr<const std::string> &buf)
    {
        size_t first_start = find_first_element_start(*buf, 0);
        size_t pos = first_start;

        auto root = parse_element(buf, pos);
        if (root)
        {
            root->prolog_start_ = 0;
            root->prolog_end_ = first_start; // **not** the advanced pos
        }
        return root;
    }

    std::string_view xmlNode::name() const noexcept
    {
        return name_new_ ? std::string_view(*name_new_) : name_view_;
    }

    std::string_view xmlNode::full() const noexcept
    {
        if (!doc_ || start_ == npos || end_ == npos)
            return {};
        return std::string_view(doc_->data() + start_, end_ - start_);
    }

    std::string_view xmlNode::content() const noexcept
    {
        if (!doc_ || content_start_ == npos || content_end_ == npos)
            return {};
        return std::string_view(doc_->data() + content_start_, content_end_ - content_start_);
    }

    const std::vector<Attr> &xmlNode::attrs() const noexcept { return attrs_; }
    std::vector<std::shared_ptr<xmlNode>> &xmlNode::children() { return children_; }

    bool xmlNode::modified(bool deep) const noexcept
    {
        if (modified_)
            return true;
        if (!deep)
            return false;
        for (auto &c : children_)
            if (c->modified(true))
                return true;
        return false;
    }

    bool xmlNode::is_leaf() const noexcept { return children_.empty(); }

    void xmlNode::set_name(std::string new_name)
    {
        name_new_ = std::move(new_name);
        modified_ = true;
    }

    void xmlNode::set_content(std::string new_content)
    {
        content_writer_ = {}; // prefer explicit content over writer
        content_new_ = std::move(new_content);
        modified_ = true;
    }

    void xmlNode::set_content_writer(std::function<void(std::ostream &)> writer)
    {
        content_new_.reset();
        content_writer_ = std::move(writer);
        modified_ = true;
    }

    bool xmlNode::set_attr(std::string_view key, std::string new_value)
    {
        for (auto &a : attrs_)
        {
            if (a.name() == key)
            {
                a.value_new = std::move(new_value);
                modified_ = true;
                return true;
            }
        }
        return false;
    }

    void xmlNode::add_attr(std::string name, std::string value)
    {
        Attr a;
        a.name_new = std::move(name); // brand-new, so these live in COW fields
        a.value_new = std::move(value);
        attrs_.push_back(std::move(a));
        modified_ = true;
    }

    void xmlNode::add_child(std::shared_ptr<xmlNode> child, bool add_nl)
    {
        auto *raw = child.get();
        children_.push_back(std::move(child));
        // Default placement: end of this node's content (before end tag)
        inserts_.push_back(InsertHint{InsertHint::Where::AtEnd, nullptr, 0, raw});
        self_closing_ = false;
        modified_ = true;
        children_.back()->append_nl_start = add_nl;
        append_nl_end = add_nl;
    }

    bool xmlNode::insert_child_before(size_t anchor_in_doc_ordinal, std::shared_ptr<xmlNode> child) noexcept
    {
        const xmlNode *anchor = nth_in_doc_child(anchor_in_doc_ordinal);
        if (!anchor)
            return false;
        auto *raw = child.get();
        children_.push_back(std::move(child));
        inserts_.push_back(InsertHint{InsertHint::Where::Before, anchor, 0, raw});
        modified_ = true;
        return true;
    }
    bool xmlNode::insert_child_after(size_t anchor_in_doc_ordinal, std::shared_ptr<xmlNode> child) noexcept
    {
        const xmlNode *anchor = nth_in_doc_child(anchor_in_doc_ordinal);
        if (!anchor)
            return false;
        auto *raw = child.get();
        children_.push_back(std::move(child));
        inserts_.push_back(InsertHint{InsertHint::Where::After, anchor, 0, raw});
        modified_ = true;
        return true;
    }
    bool xmlNode::replace_child(size_t anchor_in_doc_ordinal, std::shared_ptr<xmlNode> child) noexcept
    {
        const xmlNode *anchor = nth_in_doc_child(anchor_in_doc_ordinal);
        if (!anchor)
            return false;
        // suppress the anchor
        remove_child(anchor);
        // insert new before where the old one was
        auto *raw = child.get();
        children_.push_back(std::move(child));
        inserts_.push_back(InsertHint{InsertHint::Where::Before, anchor, 0, raw});
        modified_ = true;
        return true;
    }
    bool xmlNode::replace_child(std::string_view anchor_name, std::shared_ptr<xmlNode> child, bool add_nl) noexcept
    {
        size_t anchor_index = npos;
        for (size_t i = 0; i < children_.size(); ++i)
        {
            if (children_[i] && children_[i]->name() == anchor_name)
            {
                anchor_index = i;
                break;
            }
        }
        if (anchor_index == children_.size() || anchor_index == npos)
            this->add_child(child, add_nl); // if not found, just add to the end
        const xmlNode *anchor = nth_in_doc_child(anchor_index);
        if (!anchor)
            return false;
        // suppress the anchor
        remove_child(anchor);
        // insert new before where the old one was
        auto *raw = child.get();
        children_.push_back(std::move(child));
        inserts_.push_back(InsertHint{InsertHint::Where::Before, anchor, 0, raw});
        modified_ = true;
        children_.back()->append_nl_start = add_nl;
        append_nl_end = add_nl;
        return true;
    }
    bool xmlNode::insert_child_at_content_offset(size_t rel_offset, std::shared_ptr<xmlNode> child) noexcept
    {
        if (content_start_ == npos || content_end_ == npos)
            return false;
        size_t content_len = content_end_ - content_start_;
        if (rel_offset > content_len)
            rel_offset = content_len; // clamp
        size_t abs = content_start_ + rel_offset;
        auto *raw = child.get();
        children_.push_back(std::move(child));
        inserts_.push_back(InsertHint{InsertHint::Where::AtAbs, nullptr, abs, raw});
        modified_ = true;
        return true;
    }
    bool xmlNode::insert_child_at_start(std::shared_ptr<xmlNode> child) noexcept
    {
        auto *raw = child.get();
        children_.push_back(std::move(child));
        inserts_.push_back(InsertHint{InsertHint::Where::AtStart, nullptr, 0, raw});
        modified_ = true;
        return true;
    }
    bool xmlNode::insert_child_at_end(std::shared_ptr<xmlNode> child) noexcept
    {
        auto *raw = child.get();
        children_.push_back(std::move(child));
        inserts_.push_back(InsertHint{InsertHint::Where::AtEnd, nullptr, 0, raw});
        modified_ = true;
        return true;
    }

    // -------------------- xmlNode (private helpers) --------------------

    xmlNode::xmlNode(std::shared_ptr<const std::string> doc) : doc_(std::move(doc)) {}

    static inline bool is_space(char c)
    {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }

    size_t xmlNode::find_first_element_start(const std::string &s, size_t pos)
    {
        const size_t N = s.size();
        while (pos < N)
        {
            auto lt = s.find('<', pos);
            if (lt == npos)
                return N;
            if (lt + 1 >= N)
                return lt;
            // Skip comments
            if (lt + 3 < N && s.compare(lt, 4, "<!--") == 0)
            {
                size_t end = s.find("-->", lt + 4);
                pos = (end == npos) ? N : end + 3;
                continue;
            }
            // Skip processing instruction
            if (s.compare(lt, 2, "<?") == 0)
            {
                size_t end = s.find("?>", lt + 2);
                pos = (end == npos) ? N : end + 2;
                continue;
            }
            // Skip DOCTYPE or other declarations
            if (s.compare(lt, 2, "<!") == 0 && !(lt + 9 < N && s.compare(lt, 9, "<![CDATA[") == 0))
            {
                size_t end = s.find('>', lt + 2);
                pos = (end == npos) ? N : end + 1;
                continue;
            }
            // CDATA should be part of content, not the first element — skip it here.
            if (lt + 9 < N && s.compare(lt, 9, "<![CDATA[") == 0)
            {
                size_t end = s.find("]]>", lt + 9);
                pos = (end == npos) ? N : end + 3;
                continue;
            }
            // Real element
            return lt;
        }
        return N;
    }

    bool xmlNode::skip_comment(const std::string &s, size_t &pos)
    {
        if (pos + 3 < s.size() && s.compare(pos, 4, "<!--") == 0)
        {
            size_t end = s.find("-->", pos + 4);
            pos = (end == npos) ? s.size() : end + 3;
            return true;
        }
        return false;
    }

    bool xmlNode::skip_pi(const std::string &s, size_t &pos)
    {
        if (pos + 1 < s.size() && s.compare(pos, 2, "<?") == 0)
        {
            size_t end = s.find("?>", pos + 2);
            pos = (end == npos) ? s.size() : end + 2;
            return true;
        }
        return false;
    }

    bool xmlNode::skip_doctype(const std::string &s, size_t &pos)
    {
        if (pos + 1 < s.size() && s.compare(pos, 2, "<!") == 0 && !(pos + 9 < s.size() && s.compare(pos, 9, "<![CDATA[") == 0))
        {
            size_t end = s.find('>', pos + 2);
            pos = (end == npos) ? s.size() : end + 1;
            return true;
        }
        return false;
    }

    bool xmlNode::skip_cdata(const std::string &s, size_t &pos)
    {
        if (pos + 9 <= s.size() && s.compare(pos, 9, "<![CDATA[") == 0)
        {
            size_t end = s.find("]]>", pos + 9);
            pos = (end == npos) ? s.size() : end + 3; // advance past entire CDATA
            return true;
        }
        return false;
    }

    void xmlNode::parse_attributes(xmlNode &node, size_t &cur)
    {
        const auto &s = *node.doc_;
        const size_t N = s.size();

        auto skip_ws = [&](size_t &i)
        {
            while (i < N && is_space(s[i]))
                ++i;
        };

        while (cur < N)
        {
            skip_ws(cur);
            if (cur >= N)
                break;
            if (s[cur] == '>')
            {
                node.head_end_ = cur;
                return;
            }
            if (s[cur] == '/' && cur + 1 < N && s[cur + 1] == '>')
            {
                node.head_end_ = cur + 1;
                return;
            }

            // name
            size_t name_beg = cur;
            while (cur < N && !is_space(s[cur]) && s[cur] != '=' && s[cur] != '>' && s[cur] != '/')
                ++cur;
            size_t name_end = cur;

            skip_ws(cur);
            if (cur >= N || s[cur] != '=')
            {
                // Malformed attribute (no '='). Treat as boolean attribute with empty value.
                Attr a;
                a.name_view = std::string_view(s.data() + name_beg, name_end - name_beg);
                a.value_view = std::string_view{};
                node.attrs_.push_back(std::move(a));
                continue;
            }
            ++cur; // skip '='
            skip_ws(cur);
            if (cur >= N)
                break;

            char quote = s[cur];
            if (quote != '"' && quote != '\'')
            {
                // Unquoted value (not strictly XML). Read until ws or tag end.
                size_t val_beg = cur;
                while (cur < N && !is_space(s[cur]) && s[cur] != '>' && s[cur] != '/')
                    ++cur;
                Attr a;
                a.name_view = std::string_view(s.data() + name_beg, name_end - name_beg);
                a.value_view = std::string_view(s.data() + val_beg, cur - val_beg);
                node.attrs_.push_back(std::move(a));
                continue;
            }

            ++cur; // after opening quote
            size_t val_beg = cur;
            size_t val_end = s.find(quote, cur);
            if (val_end == npos)
            {
                val_end = N;
                cur = N;
            }
            else
            {
                cur = val_end + 1;
            }

            Attr a;
            a.name_view = std::string_view(s.data() + name_beg, name_end - name_beg);
            a.value_view = std::string_view(s.data() + val_beg, val_end - val_beg);
            node.attrs_.push_back(std::move(a));
        }
    }

    std::shared_ptr<xmlNode> xmlNode::parse_element(const std::shared_ptr<const std::string> &doc, size_t &pos)
    {
        const std::string &s = *doc;
        const size_t N = s.size();
        if (pos >= N || s[pos] != '<')
            return nullptr;

        auto node = std::shared_ptr<xmlNode>(new xmlNode(doc));
        node->start_ = pos;

        // Read name
        size_t cur = pos + 1;
        // skip any whitespace after '<'
        while (cur < N && is_space(s[cur]))
            ++cur;
        size_t name_beg = cur;
        while (cur < N && !is_space(s[cur]) && s[cur] != '>' && s[cur] != '/')
            ++cur;
        size_t name_end = cur;

        node->name_view_ = std::string_view(s.data() + name_beg, name_end - name_beg);

        // Attributes and tag end
        parse_attributes(*node, cur);

        // Detect "/>" even if there is whitespace before '/', e.g. "<x   />".
        // head_end_ points at '>' (or at the second char of "/>"), so look at the char before '>'.
        node->self_closing_ =
            (node->head_end_ != npos &&
             node->head_end_ > node->start_ &&
             (*node->doc_)[node->head_end_ - 1] == '/');

        if (node->self_closing_)
        {
            // Treat as empty content; keep offsets consistent
            node->content_start_ = node->head_end_ + 1;
            node->content_end_ = node->head_end_ + 1;
            node->end_ = node->head_end_ + 1;
            pos = node->end_;
            return node;
        }

        // Normal element with content and possibly children
        node->content_start_ = (node->head_end_ == npos) ? N : node->head_end_ + 1;

        // Scan content interleaving children until matching end tag
        size_t cursor = node->content_start_;
        while (cursor < N)
        {
            size_t lt = s.find('<', cursor);
            if (lt == npos)
            {
                // Malformed / missing end tag; consume to end
                node->content_end_ = N;
                node->end_ = N;
                pos = N;
                return node;
            }

            // Check for closing tag of this node
            if (lt + 1 < N && s[lt + 1] == '/')
            {
                node->content_end_ = lt;
                // find end of closing tag
                size_t gt = s.find('>', lt + 2);
                node->end_ = (gt == npos) ? N : gt + 1;
                pos = node->end_;
                return node;
            }

            // Skippable markup treated as content-only (not nodes here)
            size_t temp = lt;
            if (skip_comment(s, temp) || skip_pi(s, temp))
            {
                cursor = temp; // comments and PIs aren't children; content continues
                continue;
            }
            if (skip_doctype(s, temp))
            {
                cursor = temp;
                continue;
            }
            if (skip_cdata(s, temp))
            {
                cursor = temp;
                continue;
            } // CDATA becomes raw content

            // Child node
            size_t child_pos = lt;
            auto child = parse_element(doc, child_pos);
            if (!child)
            {
                // If parse failed, avoid infinite loop by moving cursor forward
                cursor = lt + 1;
                continue;
            }
            node->children_.push_back(std::move(child));
            cursor = child_pos; // parse_element advanced child_pos to end of child
        }

        // Shouldn't reach here ordinarily
        node->content_end_ = cursor;
        node->end_ = cursor;
        pos = cursor;
        return node;
    }

    // -------------------- Writer --------------------

    bool xmlNode::modified_header() const noexcept
    {
        if (name_new_.has_value())
            return true;
        for (auto const &a : attrs_)
            if (a.modified())
                return true;
        return false;
    }

    bool xmlNode::modified_footer() const noexcept
    {
        return name_new_.has_value();
    }

    void xmlNode::write_start_tag(std::ostream &os) const
    {
        if (!modified_header())
        {
            os.write(doc_->data() + start_, static_cast<std::streamsize>((head_end_ + 1) - start_));
            return;
        }

        if (append_nl_start)
            os.put('\n');

        os.put('<');
        auto nm = name();
        os.write(nm.data(), static_cast<std::streamsize>(nm.size()));
        for (auto const &a : attrs_)
        {
            os.put(' ');
            auto an = a.name();
            os.write(an.data(), static_cast<std::streamsize>(an.size()));
            os.write("=\"", 2);
            auto av = a.value();
            os.write(av.data(), static_cast<std::streamsize>(av.size()));
            os.put('"');
        }
        if (self_closing_)
        {
            os.write("/>", 2); // canonicalize to "/>" on rebuild
        }
        else
        {
            os.put('>');
        }
    }

    void xmlNode::write_end_tag(std::ostream &os) const
    {
        if (self_closing_)
            return; // self-closing elements have no end tag

        if (!modified_footer())
        {
            size_t tail_from = (content_end_ == npos) ? (head_end_ + 1) : content_end_;
            if (append_nl_end)
                os << "\n";
            os.write(doc_->data() + tail_from, static_cast<std::streamsize>(end_ - tail_from));
            return;
        }
        if (append_nl_end)
            os << "\n";
        os.write("</", 2);
        auto nm = name();
        os.write(nm.data(), static_cast<std::streamsize>(nm.size()));
        os.put('>');
    }

    void xmlNode::write(std::ostream &os) const { write_impl(os, /*as_child=*/false); }

    void xmlNode::write(std::string &out) const
    {
        struct StringAppendBuf : std::streambuf
        {
            std::string *s;
            explicit StringAppendBuf(std::string *sp) : s(sp) {}
            std::streamsize xsputn(const char *p, std::streamsize n) override
            {
                s->append(p, size_t(n));
                return n;
            }
            int overflow(int ch) override
            {
                if (ch != EOF)
                    s->push_back(char(ch));
                return ch;
            }
        } buf{&out};
        std::ostream os(&buf);
        write_impl(os, /*as_child=*/false);
    }

    void xmlNode::write_impl(std::ostream &os, bool as_child) const
    {
        // Prolog only for root
        if (!as_child && prolog_end_ > prolog_start_ && prolog_end_ <= start_)
        {
            os.write(doc_->data() + prolog_start_, static_cast<std::streamsize>(prolog_end_ - prolog_start_));
        }

        // Fast path: unmodified leaf
        if (!modified_ && children_.empty())
        {
            os.write(doc_->data() + start_, static_cast<std::streamsize>(end_ - start_));
            return;
        }

        // Start tag
        if (start_ != npos)
            write_start_tag(os);

        auto is_in_doc = [&](const std::shared_ptr<xmlNode> &c) -> bool
        {
            return c->doc_.get() == this->doc_.get() && c->start_ != npos && c->end_ != npos;
        };

        // NEW: consumed flags for all inserts_ entries (prevents double-emits & infinite loops)
        std::vector<char> consumed(inserts_.size(), 0);

        // Emit all AtStart inserts (once)
        for (size_t i = 0; i < inserts_.size(); ++i)
        {
            const auto &ins = inserts_[i];
            if (ins.where == InsertHint::Where::AtStart && !consumed[i])
            {
                ins.node->write_impl(os, /*as_child=*/true);
                consumed[i] = 1;
            }
        }

        // Content
        if (content_start_ != npos && content_end_ != npos)
        {
            if (content_writer_)
            {
                content_writer_(os);
            }
            else if (content_new_)
            {
                os.write(content_new_->data(), static_cast<std::streamsize>(content_new_->size()));
            }
            else
            {
                size_t cursor = content_start_;

                // Helper: flush unconsumed AtAbs inserts in [cursor, limit)
                auto flush_atabs_until = [&](size_t limit)
                {
                    while (true)
                    {
                        size_t min_abs = npos;
                        for (size_t i = 0; i < inserts_.size(); ++i)
                        {
                            const auto &ins = inserts_[i];
                            if (consumed[i])
                                continue;
                            if (ins.where != InsertHint::Where::AtAbs)
                                continue;
                            if (ins.abs >= cursor && ins.abs < limit)
                            {
                                if (min_abs == npos || ins.abs < min_abs)
                                    min_abs = ins.abs;
                            }
                        }
                        if (min_abs == npos)
                            break;

                        // write gap up to min_abs
                        if (cursor < min_abs)
                        {
                            os.write(doc_->data() + cursor, static_cast<std::streamsize>(min_abs - cursor));
                            cursor = min_abs;
                        }
                        // emit all AtAbs exactly at min_abs (that are not consumed)
                        for (size_t i = 0; i < inserts_.size(); ++i)
                        {
                            const auto &ins = inserts_[i];
                            if (!consumed[i] && ins.where == InsertHint::Where::AtAbs && ins.abs == min_abs)
                            {
                                ins.node->write_impl(os, /*as_child=*/true);
                                consumed[i] = 1; // mark consumed to avoid re-finding it
                            }
                        }
                        // keep cursor at min_abs; next loop will look for strictly greater abs
                    }
                };

                // Iterate over in-doc children in stored order
                for (auto const &c : children_)
                {
                    if (!is_in_doc(c))
                        continue;

                    // First flush any AtAbs inserts before this child
                    flush_atabs_until(c->start_);

                    // Write gap up to the child's start
                    if (cursor < c->start_)
                    {
                        os.write(doc_->data() + cursor, static_cast<std::streamsize>(c->start_ - cursor));
                        cursor = c->start_;
                    }

                    // Inserts anchored BEFORE this child (consume them)
                    for (size_t i = 0; i < inserts_.size(); ++i)
                    {
                        const auto &ins = inserts_[i];
                        if (!consumed[i] && ins.where == InsertHint::Where::Before && ins.anchor == c.get())
                        {
                            ins.node->write_impl(os, /*as_child=*/true);
                            consumed[i] = 1;
                        }
                    }
                    // Also flush AtAbs exactly at child's start (consume them)
                    for (size_t i = 0; i < inserts_.size(); ++i)
                    {
                        const auto &ins = inserts_[i];
                        if (!consumed[i] && ins.where == InsertHint::Where::AtAbs && ins.abs == c->start_)
                        {
                            ins.node->write_impl(os, /*as_child=*/true);
                            consumed[i] = 1;
                        }
                    }

                    // Child itself (unless suppressed)
                    if (!c->suppressed_)
                        c->write_impl(os, /*as_child=*/true);
                    cursor = c->end_;

                    // Inserts anchored AFTER this child (consume them)
                    for (size_t i = 0; i < inserts_.size(); ++i)
                    {
                        const auto &ins = inserts_[i];
                        if (!consumed[i] && ins.where == InsertHint::Where::After && ins.anchor == c.get())
                        {
                            ins.node->write_impl(os, /*as_child=*/true);
                            consumed[i] = 1;
                        }
                    }
                }

                // Flush remaining AtAbs up to content_end_ (consume)
                flush_atabs_until(content_end_);

                // Tail gap
                if (cursor < content_end_)
                {
                    os.write(doc_->data() + cursor, static_cast<std::streamsize>(content_end_ - cursor));
                    cursor = content_end_;
                }

                // Finally, AtEnd inserts (consume)
                for (size_t i = 0; i < inserts_.size(); ++i)
                {
                    const auto &ins = inserts_[i];
                    if (!consumed[i] && ins.where == InsertHint::Where::AtEnd)
                    {
                        ins.node->write_impl(os, /*as_child=*/true);
                        consumed[i] = 1;
                    }
                }
            }
        }
        // End tag
        if (end_ != npos)
            write_end_tag(os);
    }

    // Deep copy of an xmlNode that creates a new source string
    // Allows for independent manipulation of the copy,
    // and allows for storing only certain nodes in memory
    // rather than the entire document
    std::shared_ptr<xmlNode> xmlNode::deep_copy() const
    {
        std::string copy_content;
        this->write(copy_content);                // write the full node text into a string
        auto copy = xmlNode::parse(copy_content); // copy full node text, ie a copy of just the content of this node
        return copy;
    }

    bool xmlNode::has_child(std::string_view name) const noexcept
    {
        return std::any_of(children_.begin(), children_.end(),
                           [&](const auto &child)
                           { return child->name() == name; });
    }

    std::shared_ptr<xmlNode> xmlNode::get_child(std::string_view name) const noexcept
    {
        auto it = std::find_if(children_.begin(), children_.end(),
                               [&](const auto &child)
                               { return child->name() == name; });
        return (it != children_.end()) ? *it : nullptr;
    }

    std::vector<std::shared_ptr<xmlNode>> xmlNode::get_children(std::string_view name) const noexcept
    {
        std::vector<std::shared_ptr<xmlNode>> result;
        for (const auto &child : children_)
        {
            if (child->name() == name)
            {
                result.push_back(child);
            }
        }
        return result;
    }

    bool xmlNode::remove_child(size_t index) noexcept
    {
        if (index >= children_.size())
            return false;
        children_[index]->suppressed_ = true;
        modified_ = true;
        return true;
    }

    bool xmlNode::remove_child(const xmlNode *child) noexcept
    {
        for (auto &c : children_)
        {
            if (c.get() == child)
            {
                c->suppressed_ = true;
                modified_ = true;
                return true;
            }
        }
        return false;
    }

    bool xmlNode::remove_child(std::string_view name) noexcept
    {
        for (auto &c : children_)
        {
            if (c->name() == name)
            {
                c->suppressed_ = true;
                modified_ = true;
                return true;
            }
        }
        return false;
    }

    const xmlNode *xmlNode::nth_in_doc_child(size_t ordinal) const noexcept
    {
        size_t seen = 0;
        for (auto const &c : children_)
        {
            bool in_doc = (c->doc_.get() == this->doc_.get() && c->start_ != npos && c->end_ != npos);
            if (!in_doc)
                continue;
            if (seen == ordinal)
                return c.get();
            ++seen;
        }
        return nullptr;
    }

    std::string read_file(std::string_view path)
    {
        constexpr auto read_size = size_t(4096);
        auto stream = std::ifstream(path.data());
        stream.exceptions(std::ios_base::badbit);
        if (not stream)
        {
            throw std::ios_base::failure("file does not exist");
        }
        auto out = std::string();
        auto buf = std::string(read_size, '\0');
        while (stream.read(&buf[0], read_size))
        {
            out.append(buf, 0, stream.gcount());
        }
        out.append(buf, 0, stream.gcount());
        return out;
    }

    std::vector<std::string_view> line_splitter(std::string_view content)
    {
        std::vector<std::string_view> lines;
        size_t start = 0;
        size_t end = 0;
        while ((end = content.find('\n', start)) != npos)
        {
            lines.push_back(content.substr(start, end - start));
            start = end + 1;
        }
        lines.push_back(content.substr(start));
        return lines;
    }

    std::vector<std::string_view> blank_splitter(std::string_view content)
    {
        std::vector<std::string_view> words;
        size_t start = 0;
        size_t end = 0;
        while ((end = content.find_first_of(" \n\t", start)) != npos)
        {
            if (end > start)
            {
                words.push_back(content.substr(start, end - start));
            }
            start = end + 1;
        }
        if (start < content.size())
        {
            words.push_back(content.substr(start));
        }
        return words;
    }

    std::shared_ptr<event> string_to_event(std::string_view content)
    {
        auto lines = line_splitter(content);
        if (lines.empty())
            return nullptr;
        if (lines.size() < 2)
            return nullptr;
        size_t ind = 0;
        while (ind < lines.size() && lines[ind].empty())
            ++ind;
        if (lines[0].find("<event") != npos)
            ++ind;
        auto head_line = blank_splitter(lines[ind]);
        if (head_line.size() < 6)
            return nullptr;
        auto n_prt = ctoi(head_line[0]);
        if (n_prt <= 0)
            return nullptr;
        auto ev = std::make_shared<event>(n_prt);
        ev->set_proc_id(ctoi(head_line[1])).set_weight(ctod(head_line[2])).set_scale(ctod(head_line[3])).set_alphaEW(ctod(head_line[4])).set_alphaS(ctod(head_line[5]));
        ++ind;
        for (auto prt : *ev)
        {
            auto words = blank_splitter(lines[ind]);
            prt.set_pdg(ctoi(words[0])).set_status(ctoi(words[1])).set_mother(ctoi(words[2]), ctoi(words[3])).set_icol(ctoi(words[4]), ctoi(words[5])).set_momentum(ctod(words[9]), ctod(words[6]), ctod(words[7]), ctod(words[8])).set_mass(ctod(words[10])).set_vtim(ctod(words[11])).set_spin(ctod(words[12]));
            ++ind;
        }
        return ev;
    }

    std::shared_ptr<event> xml_to_event(std::shared_ptr<xmlNode> node)
    {
        if (node->name() != "event")
            return nullptr;
        auto ev = string_to_event(node->content());
        if (node->n_children() > 0)
        {
            for (auto child : node->children())
            {
                if (child->name() == "weights")
                {
                    // Handle weights child node
                    auto wgt_strings = blank_splitter(child->content());
                    for (const auto &wgt : wgt_strings)
                    {
                        ev->add_wgt(ctod(wgt));
                    }
                }
                else if (child->name() == "rwgt")
                {
                    // Handle reweight child node
                    for (auto rwgt_child : child->children())
                    {
                        if (rwgt_child->name() == "wgt")
                        {
                            ev->add_wgt(ctod(rwgt_child->content()));
                        }
                    }
                }
                else if (child->name() == "scales")
                {
                    for (auto attr : child->attrs())
                    {
                        std::string aname(attr.name_view);
                        std::transform(aname.begin(), aname.end(), aname.begin(), ::tolower);
                        if (aname == "mur")
                            ev->set_muR(ctod(attr.value_view));
                        else if (aname == "muf")
                            ev->set_muF(ctod(attr.value_view));
                        else if (aname == "mups")
                            ev->set_muPS(ctod(attr.value_view));
                    }
                }
                else
                {
                    auto copy = child->deep_copy();
                    ev->extra[std::string(copy->name())] = copy;
                }
            }
        }
        return ev;
    }

    initNode string_to_init(std::string_view content)
    {
        auto lines = line_splitter(content);
        size_t ind = 0;
        while (ind < lines.size() && lines[ind].empty())
            ++ind;
        if (lines[0].find("<init") != npos)
            ++ind;
        auto head_line = blank_splitter(lines[ind]);
        auto init = initNode();
        init.set_idBm(ctoi(head_line[0]), ctoi(head_line[1])).set_eBm(ctod(head_line[2]), ctod(head_line[3])).set_pdfG(ctoi(head_line[4]), ctoi(head_line[5])).set_pdfS(ctod(head_line[6]), ctod(head_line[7])).set_idWgt(ctoi(head_line[8])).set_nProc(ctoi(head_line[9]));
        ++ind;
        auto num_proc = init.nProcUP();
        for (size_t i = 0; i < num_proc; ++i)
        {
            auto proc_line = blank_splitter(lines[ind]);
            init.add_xSec(ctod(proc_line[0])).add_xSecErr(ctod(proc_line[1])).add_xMax(ctod(proc_line[2])).add_lProc(ctoi(proc_line[3]));
            ++ind;
        }
        return init;
    }

    initNode xml_to_init(std::shared_ptr<xmlNode> node)
    {
        auto init = string_to_init(node->content());
        if (node->n_children() > 0)
        {
            for (auto child : node->children())
            {
                auto copy = child->deep_copy();
                init.extra[std::string(copy->name())] = copy;
            }
        }
        return init;
    }

    std::any xml_to_any(std::shared_ptr<xmlNode> node)
    {
        if (!node)
            return std::nullopt;
        auto new_node = node->deep_copy();
        return std::any(new_node);
    }

    std::shared_ptr<xmlNode> init_to_xml(const initNode &init)
    {
        std::string s = write_stream(&initNode::print_init, init);
        return xmlNode::parse(s);
    }

    std::shared_ptr<xmlNode> event_to_xml(event &ev)
    {
        std::string s = write_stream(&event::print, ev, true);
        return xmlNode::parse(s);
    }

    std::optional<std::shared_ptr<xmlNode>> header_to_xml(const std::any &a)
    {
        if (!a.has_value())
            return std::nullopt;

        if (a.type() == typeid(std::shared_ptr<xmlNode>))
        {
            auto p = std::any_cast<std::shared_ptr<xmlNode>>(a);
            if (p)
                return p;
            return std::nullopt;
        }
        if (a.type() == typeid(std::string))
        {
            return xmlNode::parse(std::any_cast<const std::string &>(a));
        }
        // Unknown header payload: ignore
        return std::nullopt;
    }

    // Explicit XML instantiation definition (prebuilds the specialization)
    template class lheReader<std::shared_ptr<xmlNode>,
                             std::shared_ptr<xmlNode>,
                             std::shared_ptr<xmlNode>>;
    template class lheWriter<InitToXmlFn, EventToXmlFn, HeaderToXmlFn>;

    const xmlReader &xml_reader()
    {
        static const xmlReader b{&xml_to_init, &xml_to_event, &xml_to_any};
        return b;
    }

    const xmlWriter &xml_writer()
    {
        static const xmlWriter t{&init_to_xml, &event_to_xml, &header_to_xml};
        return t;
    }

    xmlRaw to_xml_raw(const lhe &doc)
    {
        return xml_writer().to_raw(doc);
    }

    std::shared_ptr<xmlNode> to_xml(xmlRaw &raw)
    {
        std::stringstream ss;
        ss << "<LesHouchesEvents version=\"3.0\">\n";

        if (raw.header && *raw.header) // first: has optional, second: non-null shared_ptr
        {
            (*raw.header)->write(ss);
            raw.header.reset(); // reset the optional itself
        }

        raw.init->write(ss);
        raw.init.reset();

        for (auto &event : raw.events)
        {
            event->write(ss);
            event.reset();
        }

        ss << "\n</LesHouchesEvents>";
        return xmlNode::parse(ss.str());
    }

    std::shared_ptr<xmlNode> to_xml(const lhe &doc)
    {
        auto raw = to_xml_raw(doc);
        return to_xml(raw);
    }

    lhe to_lhe(std::shared_ptr<xmlNode> node)
    {
        auto builder = xml_reader();
        auto init = node->get_child("init");
        auto events = node->get_children("event");
        auto header = std::make_optional(node->get_child("header"));
        return builder.read(init, events, header);
    }

    lhe to_lhe(const std::string &xml)
    {
        return to_lhe(xmlNode::parse(xml));
    }

    lhe load_lhef(std::istream &in)
    {
        // reads file line by line to avoid issues with large files
        // starts with reading the header (if it exists) and maps it to an xmlNode
        // then reads the init node and makes an initNode object
        // then reads each event node and makes an event object, creating a vector of shared ptrs to events
        auto events = std::vector<std::shared_ptr<event>>();
        auto content = std::string();
        std::string buf;
        while (buf.empty())
        {
            std::getline(in, buf);
        }
        if (buf.find("LesHouchesEvents") == npos)
        {
            throw std::runtime_error("load_lhef: not a valid LHEF file");
        }
        std::shared_ptr<xmlNode> header = nullptr;
        std::getline(in, buf);
        if (buf.find("header") != npos)
        {
            if (buf.find("</header") != npos || buf.find("/>") != npos)
            {
                header = xmlNode::parse(buf);
            }
            else
            {
                content += buf + "\n";
                while (std::getline(in, buf))
                {
                    content += buf + "\n";
                    if (buf.find("/header") != npos)
                        break;
                }
                header = xmlNode::parse(content);
                content.clear();
            }
        }
        while (buf.find("init") == npos)
        {
            std::getline(in, buf);
        }
        if (buf.find("</init") != npos || buf.find("/>") != npos)
        {
            throw std::runtime_error("load_lhef: malformed init block");
        }
        content += buf + "\n";
        while (std::getline(in, buf))
        {
            content += buf + "\n";
            if (buf.find("/init") != npos)
            {
                break;
            }
        }
        auto init = xml_to_init(xmlNode::parse(content));
        content.clear();
        while (std::getline(in, buf))
        {
            if (buf.find("<event") != npos)
            {
                if (!content.empty())
                {
                    auto curr_event = xml_to_event(xmlNode::parse(content));
                    if (curr_event)
                        events.push_back(curr_event);
                }
                content.clear();
                content += buf + "\n";
            }
            else if (buf.find("</LesHouchesEvents") != npos)
            {
                if (!content.empty())
                {
                    auto curr_event = xml_to_event(xmlNode::parse(content));
                    if (curr_event)
                        events.push_back(curr_event);
                }
                content.clear();
                break;
            }
            else
            {
                content += buf + "\n";
            }
        }
        lhe doc(init, events);
        if (header)
            doc.set_header(header);
        return doc;
    }

    lhe load_lhef(const std::string &filename)
    {
        auto stream = std::ifstream(filename);
        if (!stream)
            throw std::ios_base::failure("load_lhef: could not open file for reading");
        return load_lhef(stream);
    }

    void write_lhef(lhe &doc, std::ostream &out, bool include_weight_ids)
    {
        doc.print(out, include_weight_ids);
    }

    void write_lhef(lhe &doc, const std::string &filename, bool include_weight_ids)
    {
        auto stream = std::ofstream(filename);
        if (!stream)
            throw std::ios_base::failure("write_lhef: could not open file for writing");
        write_lhef(doc, stream, include_weight_ids);
    }

    std::shared_ptr<xmlNode> load_xml(const std::string &filename)
    {
        auto xml_content = read_file(filename);
        return xmlNode::parse(xml_content);
    }

    std::string slha::upper(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::toupper(c)); });
        return s;
    }
    static std::string ltrim_(std::string s)
    {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                        [](unsigned char c)
                                        { return !std::isspace(c); }));
        return s;
    }
    static std::string rtrim_(std::string s)
    {
        s.erase(std::find_if(s.rbegin(), s.rend(),
                             [](unsigned char c)
                             { return !std::isspace(c); })
                    .base(),
                s.end());
        return s;
    }
    std::string slha::trim(const std::string &s) { return rtrim_(ltrim_(s)); }

    bool slha::starts_with_ci(const std::string &s, const char *prefix)
    {
        size_t n = std::char_traits<char>::length(prefix);
        if (s.size() < n)
            return false;
        for (size_t i = 0; i < n; ++i)
        {
            if (std::toupper(static_cast<unsigned char>(s[i])) !=
                std::toupper(static_cast<unsigned char>(prefix[i])))
                return false;
        }
        return true;
    }

    std::string slha::indices_to_string(std::initializer_list<int> indices)
    {
        std::ostringstream os;
        os << '{';
        bool first = true;
        for (int v : indices)
        {
            if (!first)
                os << ',';
            os << v;
            first = false;
        }
        os << '}';
        return os.str();
    }

    void slha::read(std::istream &in)
    {
        blocks_.clear();
        decays_.clear();

        slha::BlockData *current = nullptr;

        std::string raw;
        while (std::getline(in, raw))
        {
            std::string line = trim(raw);
            if (line.empty())
                continue;

            // Strip trailing inline comment if any
            size_t hash = line.find('#');
            if (hash != npos)
                line = rtrim_(line.substr(0, hash));
            if (line.empty())
                continue;

            if (starts_with_ci(line, "BLOCK"))
            {
                std::istringstream ss(line);
                std::string word, name;
                ss >> word >> name; // "BLOCK NAME ..."
                if (name.empty())
                {
                    current = nullptr;
                    continue;
                }
                name = upper(name);
                current = &blocks_[name]; // creates if missing
                continue;
            }

            if (starts_with_ci(line, "DECAY"))
            {
                current = nullptr;
                std::istringstream ss(line);
                std::string w;
                int pid = 0;
                double width = 0.0;
                ss >> w >> pid >> width;
                if (ss)
                    decays_[pid] = width; // ignore extras
                continue;
            }

            if (current)
            {
                // Parse indices... value
                std::istringstream ss(line);
                std::vector<std::string> toks;
                for (std::string t; ss >> t;)
                    toks.push_back(t);
                if (toks.size() < 2)
                    continue; // need at least 1 idx + value

                // value is last token
                double val = 0.0;
                try
                {
                    val = std::stod(toks.back());
                }
                catch (...)
                {
                    continue;
                }

                // preceding tokens must be ints
                std::vector<int> idx;
                idx.reserve(toks.size() - 1);
                bool ok = true;
                for (size_t i = 0; i + 1 < toks.size(); ++i)
                {
                    try
                    {
                        idx.push_back(std::stoi(toks[i]));
                    }
                    catch (...)
                    {
                        ok = false;
                        break;
                    }
                }
                if (!ok || idx.empty())
                    continue;

                current->entries[std::move(idx)] = val;
            }
        }
    }

    void slha::write(std::ostream &out,
                     int value_precision,
                     bool scientific,
                     const std::string &indent) const
    {
        auto flags_backup = out.flags();
        auto prec_backup = out.precision();

        if (scientific)
            out << std::scientific;
        out << std::setprecision(value_precision);

        // Blocks in lexical order
        for (const auto &kv : blocks_)
        {
            out << "BLOCK " << kv.first << "\n";
            for (const auto &ev : kv.second.entries)
            {
                out << indent;
                const auto &idx = ev.first;
                for (size_t i = 0; i < idx.size(); ++i)
                {
                    if (i)
                        out << ' ';
                    out << idx[i];
                }
                out << ' ' << ev.second << "\n";
            }
            out << "\n";
        }

        // Decays in pid order
        for (const auto &d : decays_)
        {
            out << "DECAY " << d.first << ' ' << d.second << "\n";
        }

        out.flags(flags_backup);
        out.precision(prec_backup);
    }

    double slha::get(const std::string &block,
                     std::initializer_list<int> indices,
                     double fallback) const
    {
        auto itb = blocks_.find(upper(block));
        if (itb == blocks_.end())
            return fallback;
        std::vector<int> key(indices.begin(), indices.end());
        auto itv = itb->second.entries.find(key);
        return (itv == itb->second.entries.end()) ? fallback : itv->second;
    }

    double slha::get(const std::string &block,
                     int i1,
                     double fallback) const
    {
        return get(block, {i1}, fallback);
    }

    void slha::set(const std::string &block,
                   std::initializer_list<int> indices,
                   double value)
    {
        auto &b = blocks_[upper(block)];
        std::vector<int> key(indices.begin(), indices.end());
        b.entries[std::move(key)] = value;
    }

    void slha::set(const std::string &block,
                   int i1,
                   double value)
    {
        set(block, {i1}, value);
    }

    double slha::get_decay(int pid, double fallback) const
    {
        auto it = decays_.find(pid);
        return (it == decays_.end()) ? fallback : it->second;
    }

    void slha::set_decay(int pid, double width)
    {
        decays_[pid] = width;
    }

    bool slha::has_block(const std::string &block) const
    {
        return blocks_.count(upper(block)) > 0;
    }

    bool slha::has_entry(const std::string &block, std::initializer_list<int> indices) const
    {
        auto itb = blocks_.find(upper(block));
        if (itb == blocks_.end())
            return false;
        std::vector<int> key(indices.begin(), indices.end());
        return itb->second.entries.count(key) > 0;
    }

    slha to_slha(std::istream &in)
    {
        return slha::parse(in);
    }

    slha to_slha(const std::string &slha_text)
    {
        std::istringstream iss(slha_text);
        return to_slha(iss);
    }

    slha to_slha(std::shared_ptr<xmlNode> node)
    {
        if (!node || node->name() != "slha")
            return slha();
        std::string content;
        node->write(content);
        return to_slha(content);
    }

    slha to_slha(lhe &doc)
    {
        if (doc.header.type() == typeid(slha))
            return std::any_cast<slha>(doc.header);
        if (doc.header.type() == typeid(std::shared_ptr<xmlNode>))
        {
            auto p = std::any_cast<std::shared_ptr<xmlNode>>(doc.header);
            if (!p)
                return slha();
            auto cont = p->get_child("slha");
            return to_slha(cont);
        }
        if (doc.header.type() == typeid(std::string))
        {
            auto head = xmlNode::parse(std::any_cast<std::string>(doc.header));
            auto p = head->get_child("slha");
            return to_slha(p);
        }
        return slha();
    }

    slha load_slha(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file)
            return slha();
        return slha::parse(file);
    }

} // namespace REX
#endif
