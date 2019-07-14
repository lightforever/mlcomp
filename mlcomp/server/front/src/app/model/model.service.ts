import {Injectable} from "@angular/core";
import {BaseService} from "../base.service";
import {BaseResult, ModelAddData, ModelStartData} from "../models";
import {AppSettings} from "../app-settings";
import {catchError} from "rxjs/operators";

@Injectable({
    providedIn: 'root'
})
export class ModelService extends BaseService {
    protected collection_part: string = 'models';
    protected single_part: string = 'model';

    add(data: ModelAddData) {
        let message = `${this.constructor.name}.add`;
        let info = {
            'dag': data.dag.id,
            'name': data.name,
            'task': data.task,
            'slot': data.slot,
            'interface': data.interface
        };
        return this.http.post<BaseResult>(AppSettings.API_ENDPOINT + this.single_part + '/add', info).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }

    start(data: ModelStartData) {
        let message = `${this.constructor.name}.start`;
        let info = {
            'dag': data.dag.id,
            'slot': data.slot,
            'interface': data.interface,
            'pipe': data.pipe,
        };
        return this.http.post<BaseResult>(AppSettings.API_ENDPOINT + this.single_part + '/start', info).pipe(
            catchError(this.handleError<BaseResult>(message, new BaseResult()))
        );
    }
}