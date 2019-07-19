import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {ReportComponent} from "./report/report.component";
import {ReportsComponent} from "./reports/reports.component";
import {ReportDetailComponent} from "./report-detail/report-detail.component";
import {LayoutsComponent} from "./layouts/layouts.component";

const routes: Routes = [
    {
        path: '',
        component: ReportComponent,
        children: [
                {path: 'list', component: ReportsComponent},
                {path: 'layouts', component: LayoutsComponent},
                {path: 'report-detail/:id', component: ReportDetailComponent},
        ]
    }
];

@NgModule({
    imports: [
        RouterModule.forChild(routes)
    ],
    exports: [
        RouterModule
    ]
})
export class ReportRoutingModule {
}
