import {NgModule} from '@angular/core';
import {SharedModule} from "./shared.module";

import {AppRoutingModule} from './app-routing.module';

import {AppComponent} from './app.component';
import {
    ProjectAddDialogComponent,
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
        ModelAddDialogComponent,
        ModelStartDialogComponent,
        MessagesComponent,
        ComputerComponent,
        ModelComponent,
        LoginComponent
    ],
    entryComponents: [ProjectAddDialogComponent,
        ModelAddDialogComponent,
        ModelStartDialogComponent
    ],
    providers: [
        { provide: HTTP_INTERCEPTORS, useClass: JwtInterceptor, multi: true },
        { provide: HTTP_INTERCEPTORS, useClass: ErrorInterceptor, multi: true }
    ],
    bootstrap: [AppComponent]
})
export class AppModule {
}
