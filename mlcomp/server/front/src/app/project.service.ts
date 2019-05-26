import {Injectable} from '@angular/core';
import {BaseService} from "./base.service";
import {BaseResult, DagStopResult} from "./models";
import {AppSettings} from "./app-settings";
import {catchError} from "rxjs/operators";

@Injectable({
    providedIn: 'root'
})
export class ProjectService extends BaseService {
    protected collection_part: string = 'projects';
    protected single_part: string = 'project';


    remove(id: number) {
        let message = `${this.constructor.name}.remove`;
        return this.http.post<DagStopResult>(AppSettings.API_ENDPOINT + this.single_part + '/remove', {'id': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove_imgs(id: number) {
        let message = `${this.constructor.name}.remove_imgs`;
        return this.http.post<DagStopResult>(AppSettings.API_ENDPOINT + 'remove_imgs', {'project': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    remove_files(id: number) {
        let message = `${this.constructor.name}.remove_files`;
        return this.http.post<DagStopResult>(AppSettings.API_ENDPOINT + 'remove_files', {'project': id}).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }
}