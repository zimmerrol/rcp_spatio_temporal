%%
%% This is file `GAUBM.cls', based on the class file `IMTEKda.cls'
%%
%% Copyright (C) 2005-09 by Simon Dreher
%% Adapted to Bachelor and Master theses at the department
%% for Physics of the University of Goettingen
%%
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either
%% version 1.3 of this license or (at your option) any later
%% version. The latest version of this license is in:
%%
%%    http://www.latex-project.org/lppl.txt
%%
%% and version 1.3 or later is part of all distributions of
%% LaTeX version 2005/12/01 or later.
%%
%% This work has the LPPL maintenance status `maintained'.
%% The Current Maintainer of this work is Thomas Pruschke
%% This work consists of all files listed in README.
%%
\NeedsTeXFormat{LaTeX2e}[2005/12/01]
\ProvidesClass{GAUBM}
[2009/03/12 v1.0 GAU Vorlage für Abschlussarbeit Physik]
\newif\if@nomtotoc\@nomtotocfalse
\newif\if@englishpreamble\@englishpreambletrue
\newif\if@preamblelangdef\@preamblelangdeffalse
\newif\if@germanopt\@germanoptfalse
\newif\if@ngermanopt\@ngermanoptfalse
\DeclareOption{nomtotoc}{\@nomtotoctrue}
\DeclareOption{noenglishpreamble}%
{\@preamblelangdeftrue\@englishpreamblefalse}
\DeclareOption{englishpreamble}%
{\@preamblelangdeftrue\@englishpreambletrue}
\DeclareOption{german}%
{\PassOptionsToClass{\CurrentOption}{scrbook}\@germanopttrue}
\DeclareOption{ngerman}%
{\PassOptionsToClass{\CurrentOption}{scrbook}\@ngermanopttrue}
\newif\if@diplom\@diplomfalse
\newif\if@review\@reviewfalse
\newif\if@bachelor\@bachelortrue
\newif\if@master\@masterfalse
\newif\ifh@snothesistype\h@snothesistypetrue
\DeclareOption{bachelor}%
{\@diplomfalse\@bachelortrue\@masterfalse\h@snothesistypefalse\@reviewfalse}
\DeclareOption{master}%
{\@diplomfalse\@bachelorfalse\@mastertrue\h@snothesistypefalse\@reviewfalse}
\DeclareOption{review}%
{\@diplomfalse\@bachelorfalse\@masterfalse\h@snothesistypefalse\@reviewtrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}
\ProcessOptions\relax
\ifh@snothesistype
\@latex@warning@no@line{%
  Eine der Optionen bachelor oder master
  \MessageBreak muss angegeben werden. Nehme bachelor.%
}%
\fi
\LoadClass[12pt,a4paper,pagesize, openany]{scrbook}
\AtBeginDocument{\RequirePackage{graphicx}\RequirePackage{datenumber}}
\RequirePackage{calc}
\def\nomname{Nomenclature}
\AfterPackage*{babel}{
  \iflanguage{german}{\@englishpreamblefalse}%
  {\if@preamblelangdef\else\@englishpreambletrue\fi}
  \addto{\captionsgerman}{\renewcommand*{\figurename}{Abb.}}
  \addto{\captionsgerman}{\renewcommand*{\tablename}{Tab.}}
  \addto{\captionsgerman}{\def\nomname{Nomenklatur}}
  \iflanguage{ngerman}{\@englishpreamblefalse}%
  {\if@preamblelangdef\else\@englishpreambletrue\fi}
  \addto{\captionsngerman}{\renewcommand*{\figurename}{Abb.}}
  \addto{\captionsngerman}{\renewcommand*{\tablename}{Tab.}}
  \addto{\captionsngerman}{\def\nomname{Nomenklatur}}
}
\AfterPackage*{german}{
  \@englishpreamblefalse\@germanopttrue
  \renewcommand*{\figurename}{Abb.}
  \renewcommand*{\tablename}{Tab.}
  \newcommand*{\captionsgermansav}{}
  \let\captionsgermansav\captionsgerman
  \renewcommand*{\captionsgerman}%
  {\captionsgermansav\def\figurename{Abb.}%
    \def\tablename{Tab.}\def\nomname{Nomenklatur}}
}
\AfterPackage*{ngerman}{
  \@englishpreamblefalse\@ngermanopttrue
  \renewcommand*{\figurename}{Abb.}
  \renewcommand*{\tablename}{Tab.}
  \newcommand*{\captionsngermansav}{}
  \let\captionsngermansav\captionsngerman
  \renewcommand*{\captionsngerman}%
  {\captionsngermansav\def\figurename{Abb.}%
    \def\tablename{Tab.}\def\nomname{Nomenklatur}}
}
%\def\FirstReferee#1{\gdef\@FirstReferee{#1}}
\newif\if@ExtraSupervisor\@ExtraSupervisorfalse
\def\@FirstRefereeWithoutOpt#1{\gdef\@FirstReferee{#1}}
\def\@FirstRefereeWithOpt[#1]#2{\gdef\@FirstReferee{#2}\@ExtraSupervisortrue%
\gdef\@ExtraSupervisor{#1}}
\def\FirstReferee{\futurelet\@FirstRefereeOpt\@FirstRefereeDecide}
\def\@FirstRefereeDecide{\ifx\@FirstRefereeOpt [ \let\@FirstRefereeNext=\@FirstRefereeWithOpt
\else  \let\@FirstRefereeNext=\@FirstRefereeWithoutOpt \fi \@FirstRefereeNext}
\def\@FirstReferee{\@latex@error{No \noexpand\FirstReferee given}\@ehc}
\def\Institute#1{\gdef\@Institute{#1}}
\def\@Institute{\@latex@error{No \noexpand\Institute given}\@ehc}
\def\SecondReferee#1{\gdef\@SecondReferee{#1}}
\def\@SecondReferee{\@latex@error{No \noexpand\SecondReferee given}\@ehc}
\def\thesistime#1{\gdef\@thesistime{#1}}
\def\@thesistime{\@latex@error{No \noexpand\thesistime given}\@ehc}
\def\ThesisAuthor#1#2{\author{#1 #2}}
\def\ThesisTitle#1#2{\title{#1}\gdef\@titleOther{#2}}
\def\@titleOther{\@latex@error{No \noexpand\titleOther given}\@ehc}
\newcounter{@countbegin}
\newcounter{@countend}
\def\ThesisBegin#1#2#3{\setmydatenumber{@countbegin}{#3}{#2}{#1}}
\def\ThesisEnd#1#2#3{\setmydatenumber{@countend}{#3}{#2}{#1}}
%\newenvironment{abstract}{\addchap*{\abstractname}}{}
\newenvironment{abstract}{\addsec*{\abstractname}}{}
\newenvironment{nomenclature}{%
  \if@nomtotoc
  \addchap{\nomname}
  \@mkboth{\nomname}{\nomname}
  \else
  \if@hyperref
  \if@openright\cleardoublepage\else\clearpage\fi
  \phantomsection\pdfbookmark{\nomname}{nom}%
  \fi
  \chapter*{\nomname}
  \@mkboth{\nomname}{\nomname}
  \fi
}{%
}
%%
%% Die Erklaerung ueber die Hilfsmittel etc. MUSS in Deutsch erfolgen.
%% Eine Aenderung der Erklaerung wird als Betrugsversuch gewertet.
%%
\newcommand{\Declaration}{%
  \clearpage\thispagestyle{empty}
  \null\vfill
  \noindent
  \begin{tabular}[t]{p{0.175\textwidth}p{0.825\textwidth-4\tabcolsep}}
    \bfseries\large Erkl\"arung&nach
    \if@bachelor\S13(8)%
    \else\S18(8)\fi{}  der Prüfungsordnung für den
    Bachelor-Studiengang Physik und den Master-Studiengang Physik
    an der Universität Göttingen:\\[1em]
    &Hiermit erkläre ich, dass ich diese Abschlussarbeit
    selbständig verfasst habe, keine anderen als die
    angegebenen Quellen und Hilfsmittel benutzt habe und alle Stellen,
    die wörtlich oder sinngemä\ss{} aus veröffentlichten Schriften
    entnommen wurden, als solche kenntlich gemacht habe.

    Darüberhinaus erkläre ich, dass diese Abschlussarbeit nicht, auch nicht
    auszugsweise, im Rahmen einer nichtbestandenen Prüfung an dieser oder
    einer anderen Hochschule eingereicht wurde.\\[1em]
    &\begin{center}Göttingen, den \today\end{center}\\[1.5cm]
    &\begin{center}(\@ThesisAuthor)\end{center}
    %\begin{minipage}[t]{0.76\textwidth-4\tabcolsep}
    %  \@author\end{minipage}
  \end{tabular}}
\def\PlaceOfBirth#1{\gdef\@WhereFrom{#1}}
\def\@WhereFrom{\@latex@error{No \noexpand\WhereFrom given}\@ehc}
\newcommand{\LLogo}{\parbox[b][2.2cm]{0.3\textwidth}{%
    \texttt{figures/Logo_Uni_Goettingen.*}
    von \texttt{http://www.physik.uni-goettingen.de/...} laden}}
\newcommand{\RLogo}{\parbox[b][2.2cm]{0.3\textwidth}{%
    \texttt{figures/Logo_Physik.*}
    von \texttt{http://www.physik.uni-goettingen.de/...} downloaden}}
\newcommand{\l@g@error}{%
  \ClassError{GAUBM}{Logo file not found}
  {One or more of the files\MessageBreak
    \space\space\space\space figures/Logo_Uni_Goettingen.eps \MessageBreak
    \space\space\space\space figures/Logo_Uni_Goettingen.pdf \MessageBreak
    \space\space\space\space figures/Logo_Physik_links.eps \MessageBreak
    \space\space\space\space figures/Logo_Physik_links.pdf \MessageBreak
    are not installed properly. \MessageBreak
    Install this subdirectory with the logo files \MessageBreak
    together with the class file, see README.\MessageBreak
    Type <return> to proceed without the logos.}
}
\IfFileExists{figures/Logo_Uni_Goettingen.eps}%
{\IfFileExists{figures/Logo_Uni_Goettingen.pdf}%
  {\renewcommand{\LLogo}%
    {\includegraphics[height=1.5cm]{figures/Logo_Uni_Goettingen}}}%
  {\l@g@error}%
}{\l@g@error}
\IfFileExists{figures/Logo_Physik_links.eps}%
{\IfFileExists{figures/Logo_Physik_links.pdf}%
  {\renewcommand{\RLogo}%
    {\includegraphics[height=1.5cm]{figures/Logo_Physik_links}}}%
  {\l@g@error}%
}{\l@g@error}
\titlehead{\LLogo\hfill\RLogo}
\def\s@bject{%
  \if@review\if@englishpreamble{Review}\else{Bericht}\fi%
  \else
  \if@bachelor\if@englishpreamble{Bachelor's Thesis}\else{Bachelorarbeit}\fi%
  \else
  \if@englishpreamble{Master's Thesis}\else{Masterarbeit}\fi%
  \fi
  \fi}
\subject{\titlefont{\s@bject}}
\date{}
\renewcommand*\maketitle[1][1]{\begin{titlepage}%
    \enlargethispage{2cm}
    \setcounter{page}{#1}
    \let\footnotesize\small
    \let\footnoterule\relax
    \let\footnote\thanks
    \renewcommand*\thefootnote{\@fnsymbol\c@footnote}%
    \let\@oldmakefnmark\@makefnmark
    \renewcommand*{\@makefnmark}{\rlap\@oldmakefnmark}
    \ifx\@titlehead\@empty \else
    \noindent\begin{minipage}[t]{\textwidth}
      \@titlehead
    \end{minipage}\par
    \fi
    \null\vfill
    \begin{center}
      \ifx\@subject\@empty \else
      {\Large \@subject \par}
      \vskip 3em
      \fi
      {\titlefont\LARGE \@title\par}
      \vskip 3em
      {\titlefont\LARGE \@titleOther\par}
      \vskip 3em
    \end{center}\par
    \noindent%
    \if@englishpreamble
    \vfil
    \noindent
    \begin{center}
      prepared by\\[3mm]
      \textbf{\large \@author}\\[3mm]
      from \@WhereFrom\\[5mm]
      at the \@Institute
    \end{center}
    \vfill\null
    \noindent
    \begin{tabular}[t]{p{0.24\linewidth}p{0.76\textwidth-4\tabcolsep}}
      \bfseries Thesis period:&
      \begin{minipage}[t]{0.76\textwidth-4\tabcolsep}%
        \dateselectlanguage{english}\setdatebynumber{\the@countbegin}\datedate\
        until \setdatebynumber{\the@countend}\datedate\end{minipage}\\&\\
        \if@ExtraSupervisor\bfseries Supervisor:&
        \begin{minipage}[t]{0.76\textwidth-4\tabcolsep}%
          \@ExtraSupervisor\end{minipage}\\&\\\fi
      \bfseries First Rreferee:&
      \begin{minipage}[t]{0.76\textwidth-4\tabcolsep}%
        \@FirstReferee\end{minipage}\\&\\
      \bfseries Second referee:&
      \begin{minipage}[t]{0.76\textwidth-4\tabcolsep}%
        \@SecondReferee\end{minipage}\\
    \end{tabular}
    \else
    {\if@ngermanopt\selectlanguage{ngerman}%
      \else\if@germanopt\selectlanguage{german}%
      \fi%
      \fi%
      \noindent%
      \noindent
      \begin{center}
        angefertigt von\\[3mm]
        \textbf{\large \@author}\\[3mm]
        aus \@WhereFrom\\[5mm]
        am \@Institute{}
      \end{center}
      \vfill\null
      \noindent
      \begin{tabular}[t]{p{0.24\linewidth}p{0.76\textwidth-4\tabcolsep}}
        \bfseries Bearbeitungszeit:&
        \begin{minipage}[t]{0.76\textwidth-4\tabcolsep}%
          \dateselectlanguage{ngerman}\setdatebynumber{\the@countbegin}\datedate\
          bis \setdatebynumber{\the@countend}\datedate\end{minipage}\\&\\
        \if@ExtraSupervisor\bfseries Betreuer/in:&
        \begin{minipage}[t]{0.76\textwidth-4\tabcolsep}%
          \@ExtraSupervisor\end{minipage}\\&\\\fi
        \bfseries Erstgutachter/in:&
        \begin{minipage}[t]{0.76\textwidth-4\tabcolsep}%
          \@FirstReferee\end{minipage}\\&\\
        \if@review
        \else
        \bfseries Zweitgutachter/in:&
        \begin{minipage}[t]{0.76\textwidth-4\tabcolsep}%
          \@SecondReferee\end{minipage}\\
        \fi%
      \end{tabular}
    }
    \fi
    \ifx\@dedication\@empty \else
    \next@tpage\null\vfill
    {\centering \Large \@dedication \par}
    \vskip \z@ \@plus3fill
    \if@twoside \next@tpage\cleardoublepage \fi
    \fi
  \end{titlepage}
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@ThesisAuthor\@author
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\@titlehead\@empty
  \global\let\@subject\@empty
  \global\let\@publishers\@empty
  \global\let\@uppertitleback\@empty
  \global\let\@lowertitleback\@empty
  \global\let\@dedication\@empty
  \global\let\author\relax
  \global\let\title\relax
  \global\let\extratitle\relax
  \global\let\titlehead\relax
  \global\let\subject\relax
  \global\let\publishers\relax
  \global\let\uppertitleback\relax
  \global\let\lowertitleback\relax
  \global\let\dedication\relax
  \global\let\date\relax
  \global\let\and\relax}
\newif\if@hyperref\@hyperreffalse
\AfterPackage*{hyperref}{%
  \@hyperreftrue
  \newcommand{\org@maketitle}{}%
  \let\org@maketitle\maketitle
  \def\maketitle{%
    \hypersetup{
      pdftitle={\@title},
      pdfauthor={\@author},
      pdfsubject={\s@bject}
    }%
    \org@maketitle
  }
}
\endinput
%%
%% End of file `GAUBM.cls'.
