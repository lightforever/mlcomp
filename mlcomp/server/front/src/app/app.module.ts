import {NgModule} from '@angular/core';
import {SharedModule} from "./shared.module";

import {AppRoutingModule} from './app-routing.module';

import {AppComponent} from './app.component';
import {
    ProjectComponent
} from './project/project.component';
import {ComputerComponent} from './computer/computer.component';
import {MessagesComponent} from './messages/messages.component';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {LoginComponent} from "./login/login.component";
import {ReactiveFormsModule} from "@angular/forms";
import {JwtInterceptor} from "./_helpers/jwt.intercepter";
import {ErrorInterceptor} from "./_helpers/error.interceptor";
import {HTTP_INTERCEPTORS} from "@angular/common/http";
import {ModelStartDialogComponent} from "./model/model-start-dialog.component";
import {ModelComponent} from "./model/model.component";
import {ModelAddDialogComponent} from "./model/model-add-dialog.component";
import {ProjectAddDialogComponent} from "./project/project-add-dialog";
import {ReportAddDialogComponent} from "./report/reports/report-add-dialog";
import {LayoutAddDialogComponent} from "./report/layouts/layout-add-dialog";
import {AuxiliaryComponent} from "./auxiliary/auxiliary.component";
import {TaskInfoDialogComponent} from "./task/task-table/task-info-dialog.component";
import {ReportUpdateDialogComponent} from "./report/report-detail/report-update-dialog.component";
import {SyncDialogComponent} from "./computer/sync-dialog";
import {AuxiliarySupervisorComponent} from "./auxiliary/supervisor/supervisor.component";

@NgModule({
    imports: [
        AppRoutingModule,
        SharedModule,
        BrowserAnimationsModule,
        ReactiveFormsModule
    ],
    declarations: [
        AppComponent,
        ProjectComponent,
        ProjectAddDialogComponent,
        ReportAddDialogComponent,
        LayoutAddDialogComponent,
        ModelAddDialogComponent,
        ModelStartDialogComponent,
        TaskInfoDialogComponent,
        ReportUpdateDialogComponent,
        MessagesComponent,
        ComputerComponent,
        ModelComponent,
        LoginComponent,
        AuxiliaryComponent,
        SyncDialogComponent,
        AuxiliarySupervisorComponent
    ],
    entryComponents: [
        ProjectAddDialogComponent,
        ModelAddDialogComponent,
        ModelStartDialogComponent,
        ReportAddDialogComponent,
        LayoutAddDialogComponent,
        TaskInfoDialogComponent,
        ReportUpdateDialogComponent,
        SyncDialogComponent
    ],
    providers: [
        {
            provide: HTTP_INTERCEPTORS,
            useClass: JwtInterceptor,
            multi: true
        },
        {
            provide: HTTP_INTERCEPTORS,
            useClass: ErrorInterceptor,
            multi: true
        }
    ],
    bootstrap: [AppComponent]
})
export class AppModule {
}
